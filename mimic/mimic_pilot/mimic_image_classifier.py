import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ── Paths ──────────────────────────────────────────────────────
# Robust path resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
if os.path.basename(CURRENT_DIR) == 'mimic_pilot':
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
else:
    BASE_DIR = os.path.abspath(CURRENT_DIR)

DATASET_CSV = os.path.join(BASE_DIR, 'mimic_pilot', 'mimic_dataset.csv')
SAVE_DIR    = os.path.join(BASE_DIR, 'mimic_pilot', 'outputs')
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 50)
print('  MIMIC-CXR Image Classifier')
print('=' * 50)
print(f'  Device : {device}')
if torch.cuda.is_available():
    print(f'  GPU    : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'  CSV    : {DATASET_CSV}')

# ── Load & Explore Dataset ─────────────────────────────────────
df = pd.read_csv(DATASET_CSV)

print(f'Total samples : {len(df)}')
print(f'Pneumonia     : {(df.label==1).sum()}')
print(f'Normal        : {(df.label==0).sum()}')
print(f'Positive rate : {(df.label==1).mean()*100:.1f}%')
print()
print(df.head(10).to_string(index=False))

# Show sample images from each class
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('MIMIC-CXR Sample Images', fontsize=15, fontweight='bold')

for col, label_name in enumerate(['Normal', 'Pneumonia']):
    subset = df[df.label_name == label_name].head(4)
    for row, (_, rec) in enumerate(subset.iterrows()):
        ax = axes[col][row]
        img = Image.open(rec.image_path).convert('L')
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{label_name}\nstudy {rec.study_id}', fontsize=8)
        ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'sample_images.png'), dpi=120)
plt.close()

# ── Dataset Class & Transforms ─────────────────────────────────
IMG_SIZE = 224

# Aggressive augmentation for small dataset
train_tfm = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=12),
    T.ColorJitter(brightness=0.25, contrast=0.25),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229]),
])

val_tfm = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485], std=[0.229]),
])

class MIMICDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row.image_path)
        label = int(row.label)
        if self.transform:
            img = self.transform(img)
        return img, label

print('Transforms ready.')
print(f'  Train: Resize({IMG_SIZE+24}) -> RandomCrop({IMG_SIZE}) -> Flip -> Rotate±12 -> Jitter -> Normalize')
print(f'  Val  : Resize({IMG_SIZE}) -> Normalize')

# ── Model (PneumoFusionNet from V3 Baseline) ───────────────────
class GCSA(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid     = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        B, C, H, W = x.shape
        avg = self.avg_pool(x).view(B, C)
        mx  = self.max_pool(x).view(B, C)
        ch_att = self.sigmoid(self.mlp(avg) + self.mlp(mx)).view(B, C, 1, 1)
        x = x * ch_att
        sp = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        return x * self.sigmoid(self.conv_spatial(sp))

class DSC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch,  3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)

class PneumoFusionNet(nn.Module):
    def __init__(self, num_classes=2, freeze_until=6):
        super().__init__()

        # ResNet50 pretrained backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 3-ch -> 1-ch: average RGB pretrained weights
        resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            resnet.conv1.weight = nn.Parameter(
                resnet.conv1.weight.mean(dim=1, keepdim=True)
            )

        # Remove final avgpool + FC
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze first N layers
        for i, child in enumerate(self.backbone.children()):
            if i < freeze_until:
                for p in child.parameters(): p.requires_grad = False

        # Paper components
        self.dsc  = DSC(2048, 1024)
        self.gcsa = GCSA(1024)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)         # [B, 2048, 7, 7]
        x = self.dsc(x)              # [B, 1024, 7, 7]
        x = self.gcsa(x)             # [B, 1024, 7, 7]
        x = self.pool(x).flatten(1)  # [B, 1024]
        return self.head(x)          # [B, 2]

    def get_features(self, x):
        x = self.backbone(x)
        x = self.dsc(x)
        x = self.gcsa(x)
        return self.pool(x).flatten(1)

# Quick sanity check
model = PneumoFusionNet().to(device)
dummy = torch.randn(4, 1, 224, 224).to(device)
print('Output shape   :', model(dummy).shape)          # [4, 2]
print('Feature shape  :', model.get_features(dummy).shape)  # [4, 1024]

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params — total : {total:,}  |  trainable: {trainable:,} ({trainable/total*100:.1f}%)')

# ── Class Weights & Training Config ────────────────────────────
BATCH_SIZE   = 8
NUM_EPOCHS   = 40
LR           = 1e-4
WEIGHT_DECAY = 1e-4
N_FOLDS      = 5

# Class weights (inverse frequency)
n0 = (df.label == 0).sum()   # Normal count
n1 = (df.label == 1).sum()   # Pneumonia count
N  = len(df)

w0 = N / (2 * n0)            # weight for Normal
w1 = N / (2 * n1)            # weight for Pneumonia
class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

print('Training Configuration')
print(f'  Batch size  : {BATCH_SIZE}')
print(f'  Epochs/fold : {NUM_EPOCHS}')
print(f'  LR          : {LR}')
print(f'  Folds       : {N_FOLDS}')
print()
print(f'Class weights  (21 Pneumonia, 118 Normal):')
print(f'  Normal    w = {w0:.3f}')
print(f'  Pneumonia w = {w1:.3f}  <- upweighted to compensate imbalance')

# ── Train & Evaluate Functions ─────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (model(imgs).argmax(1) == labels).sum().item()
        n          += len(labels)
    return total_loss / n, correct / n

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, preds_all, labels_all, probs_all = 0.0, [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss  += criterion(logits, labels).item() * len(labels)
        probs_all   += F.softmax(logits, -1)[:, 1].cpu().tolist()
        preds_all   += logits.argmax(1).cpu().tolist()
        labels_all  += labels.cpu().tolist()

    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, zero_division=0)
    try:
        auc = roc_auc_score(labels_all, probs_all)
    except ValueError:
        auc = float('nan')

    return total_loss / len(labels_all), acc, f1, auc, preds_all, labels_all, probs_all

print('Train / eval functions ready.')

# ── 5-Fold Cross-Validation Training ───────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
X, y = df.index.values, df.label.values
fold_results = []

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
    print(f'\n{"─"*50}')
    print(f'  Fold {fold+1}/{N_FOLDS}  |  Train {len(tr_idx)}  |  Val {len(vl_idx)}')
    tr_df, vl_df = df.iloc[tr_idx], df.iloc[vl_idx]
    print(f'  Train → Pneumonia:{tr_df.label.sum()}  Normal:{(tr_df.label==0).sum()}')
    print(f'  Val   → Pneumonia:{vl_df.label.sum()}  Normal:{(vl_df.label==0).sum()}')
    print(f'  {"─"*50}')

    # Weighted sampler (oversample Pneumonia in training)
    sample_w = [w1 if l == 1 else w0 for l in tr_df.label]
    sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    tr_loader = DataLoader(MIMICDataset(tr_df, train_tfm), BATCH_SIZE, sampler=sampler)
    vl_loader = DataLoader(MIMICDataset(vl_df, val_tfm),   BATCH_SIZE, shuffle=False)

    model     = PneumoFusionNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_f1, best_auc = 0.0, 0.0
    history = {k: [] for k in ['tr_loss','vl_loss','acc','f1','auc']}

    for ep in range(NUM_EPOCHS):
        tr_loss, _                          = train_epoch(model, tr_loader, criterion, optimizer)
        vl_loss, acc, f1, auc, p, l, prob  = eval_epoch(model, vl_loader, criterion)
        scheduler.step()

        history['tr_loss'].append(tr_loss)
        history['vl_loss'].append(vl_loss)
        history['acc'].append(acc)
        history['f1'].append(f1)
        history['auc'].append(auc if not np.isnan(auc) else 0)

        if f1 >= best_f1:
            best_f1, best_auc = f1, auc
            best_preds, best_labels = p, l
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f'fold{fold+1}_best.pth'))

        if (ep + 1) % 10 == 0:
            print(f'  Ep {ep+1:2d}/{NUM_EPOCHS} | Loss {tr_loss:.3f}/{vl_loss:.3f} | '
                  f'Acc {acc:.3f} | F1 {f1:.3f} | AUC {auc:.3f}')

    fold_results.append({'fold': fold+1, 'best_f1': best_f1, 'best_auc': best_auc,
                         'preds': best_preds, 'labels': best_labels, 'history': history})
    print(f'  Best  F1={best_f1:.4f}  AUC={best_auc:.4f}')

print('\n' + '='*50)
print('  TRAINING COMPLETE')
print('='*50)

# ── Results & Metrics ──────────────────────────────────────────
res = pd.DataFrame([{'Fold': r['fold'], 'F1': r['best_f1'], 'AUC': r['best_auc']}
                    for r in fold_results])
res.loc['Mean'] = res.mean()
res.loc['Std']  = res.std()
print(res.round(4).to_string())
print()
print(f"Mean F1  : {res.loc['Mean','F1']:.4f} +/- {res.loc['Std','F1']:.4f}")
print(f"Mean AUC : {res.loc['Mean','AUC']:.4f} +/- {res.loc['Std','AUC']:.4f}")
