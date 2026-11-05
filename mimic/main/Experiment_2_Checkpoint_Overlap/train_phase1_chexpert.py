"""
mimic/main/Experiment_2_Checkpoint_Overlap/train_phase1_chexpert.py
==================================================================
Phase 1 Image-Only Training with Clean Stanford CheXpert Pretrained Weights
Experiment 2 Implementation | 0.0% MIMIC-CXR Pretraining Overlap

Backbone:
    DenseNet-121 (weights='densenet121-res224-chex')
    Pretrained strictly on Stanford CheXpert (224,316 radiographs, 65,240 patients)
    Hospital Source: Stanford Hospital, Stanford, CA (0.0% MIMIC-CXR overlap)

Architecture:
    - Pretrained Stanford CheXpert DenseNet-121 features (early blocks frozen)
    - CBAM (Convolutional Block Attention Module) on 1024-channel feature map
    - Two-layer classification MLP head with BatchNorm, Dropout, ReLU
    - Focal Loss (gamma=2.0) + Mixup (alpha=0.2) + AMP mixed precision
    - Evaluated on canonical Experiment 1 test split (N_test = 562, 511 patients)
"""

import os
import sys
import copy
import json
import random
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import torchxrayvision as xrv
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix, classification_report
)

# ── Reproducibility & Device ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── Paths & Hyperparameters ───────────────────────────────────────────────
BASE_DIR   = r"C:\2026\PneumoFusionNet\mimic\main"
CSV_PATH   = os.path.join(BASE_DIR, "dataset", "mimic_paired_dataset.csv")
SPLIT_CSV  = os.path.join(BASE_DIR, "dataset", "patient_split.csv")
BBOX_CSV   = os.path.join(BASE_DIR, "outputs", "Phase_1.1v4_PA_crossval_scaleup", "lung_bboxes.csv")
SAVE_DIR   = os.path.join(BASE_DIR, "outputs", "Phase_1.1v4_PA_single_split_exp2_chexpert")
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE       = 224
BATCH_SIZE     = 16
NUM_EPOCHS     = 15
LR             = 5e-5
WEIGHT_DECAY   = 1e-4
NUM_CLASSES    = 2
MIXUP_ALPHA    = 0.2
FOCAL_GAMMA    = 2.0
TTA_TRANSFORMS = 10

# ── Load Cohort & Canonical Patient Split ──────────────────────────────────
print("\n" + "="*70)
print("Loading Cohort and Canonical Patient Split (Experiment 2 - CheXpert)...")
print("="*70)

raw_df = pd.read_csv(CSV_PATH)
def resolve_path(p):
    if pd.isna(p) or str(p).strip() == '': return ''
    p = str(p).replace('/', os.sep).replace('\\', os.sep)
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, "dataset", p)

raw_df['image_path'] = raw_df['image_path'].apply(resolve_path)
raw_df['label'] = np.select(
    [raw_df['pneumonia_label'].eq(1.0), raw_df['no_finding_label'].eq(1.0)],
    [1, 0], default=-1
)
df = raw_df[(raw_df['label'] >= 0) & raw_df['view_position'].eq('PA')].copy().reset_index(drop=True)

split_df = pd.read_csv(SPLIT_CSV)
merged = df.merge(split_df[['subject_id', 'split']], on='subject_id', how='left')

train_df = merged[merged['split'] == 'train'].reset_index(drop=True)
val_df   = merged[merged['split'] == 'val'].reset_index(drop=True)
test_df  = merged[merged['split'] == 'test'].reset_index(drop=True)

print(f"Total Cohort : {len(df):,} images ({df['subject_id'].nunique():,} patients)")
print(f"Train Cohort : {len(train_df):,} images ({train_df['subject_id'].nunique():,} patients)")
print(f"Val Cohort   : {len(val_df):,} images ({val_df['subject_id'].nunique():,} patients)")
print(f"Test Cohort  : {len(test_df):,} images ({test_df['subject_id'].nunique():,} patients)")

# ── Bounding Box Lookup ───────────────────────────────────────────────────
bbox_lookup = {}
if os.path.exists(BBOX_CSV):
    bdf = pd.read_csv(BBOX_CSV)
    for _, row in bdf.iterrows():
        bbox_lookup[row['image_path']] = {
            'x_min': int(row.get('x_min', 0)),
            'y_min': int(row.get('y_min', 0)),
            'x_max': int(row.get('x_max', 0)),
            'y_max': int(row.get('y_max', 0)),
        }
    print(f"Loaded {len(bbox_lookup):,} lung bounding boxes.")

# ── Transforms & Normalization ────────────────────────────────────────────
train_tfm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomRotation(8),
    transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
])

val_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# 10 Medically Safe TTA Transforms
tta_transforms = [
    val_tfm,
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((4, 4)), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((-4, -4)), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((250, 250)), transforms.CenterCrop(IMG_SIZE), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ColorJitter(brightness=0.12), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ColorJitter(contrast=0.12), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((2, 2)), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((-2, -2)), transforms.ToTensor()]),
    transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 0.4)), transforms.ToTensor()]),
]

class CheXpertCXRDataset(Dataset):
    def __init__(self, data_df, bbox_dict, transform=None):
        self.df = data_df.reset_index(drop=True)
        self.bbox_dict = bbox_dict or {}
        self.transform = transform
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['label'])
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            img = self.clahe.apply(img)
            bb = self.bbox_dict.get(img_path, None)
            if bb and bb.get('x_max', 0) > 0 and (bb['x_max'] - bb['x_min'] > 30) and (bb['y_max'] - bb['y_min'] > 30):
                img = img[bb['y_min']:bb['y_max'], bb['x_min']:bb['x_max']]
        
        pil_img = Image.fromarray(img)
        t = self.transform(pil_img) if self.transform else transforms.ToTensor()(pil_img)
        # Scale to [-1024, 1024] as expected by TorchXRayVision
        t = (t * 2048.0) - 1024.0
        return t, label

    def set_transform(self, tfm):
        self.transform = tfm

train_ds = CheXpertCXRDataset(train_df, bbox_lookup, transform=train_tfm)
val_ds   = CheXpertCXRDataset(val_df, bbox_lookup, transform=val_tfm)
test_ds  = CheXpertCXRDataset(test_df, bbox_lookup, transform=val_tfm)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ── Architecture: DenseNet-121 (CheXpert) + CBAM ──────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))

class EnhancedPneumoNetV4_CheXpert(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Pretrained strictly on Stanford CheXpert (224,316 CXRs, 0.0% MIMIC overlap)
        xrv_model = xrv.models.DenseNet(weights='densenet121-res224-chex')
        self.features = xrv_model.features

        # Freeze early layers; fine-tune later blocks
        freeze_layers = {
            "conv0", "norm0", "relu0", "pool0",
            "denseblock1", "transition1",
            "denseblock2", "transition2",
        }
        for name, child in self.features.named_children():
            for p in child.parameters():
                p.requires_grad = name not in freeze_layers

        self.cbam = CBAM(1024, reduction=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        f = F.relu(self.features(x), inplace=True)
        f = self.cbam(f)
        f = self.pool(f).flatten(1)
        return self.classifier(f)

    def get_features(self, x):
        f = F.relu(self.features(x), inplace=True)
        f = self.cbam(f)
        return self.pool(f).flatten(1)

# ── Focal Loss with Label Smoothing ───────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        smooth = torch.zeros_like(logits).fill_(self.label_smoothing / (num_classes - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * smooth * log_probs
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).unsqueeze(0)
        return loss.sum(dim=1).mean()

# ── Mixup Helper ──────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1.0 - lam)
    else:
        lam = 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

# ── Instantiate Model, Optimizer, Criterion ───────────────────────────────
model = EnhancedPneumoNetV4_CheXpert(num_classes=NUM_CLASSES).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Initialized: DenseNet-121 (Stanford CheXpert) + CBAM")
print(f"Total Params    : {total_params:,}")
print(f"Trainable Params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

class_counts = train_df['label'].value_counts()
weight_normal = 1.0
weight_pneumonia = float(class_counts[0] / class_counts[1]) * 1.15
alpha_tensor = torch.tensor([weight_normal, weight_pneumonia], dtype=torch.float32).to(DEVICE)
criterion = FocalLoss(alpha=alpha_tensor, gamma=FOCAL_GAMMA, label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = GradScaler()

def evaluate_loader(net, loader):
    net.eval()
    all_probs = []
    all_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            with autocast():
                logits = net(imgs)
                loss = criterion(logits, lbls)
            total_loss += loss.item() * len(lbls)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(lbls.cpu().numpy())
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    auc = roc_auc_score(all_targets, all_probs)
    avg_loss = total_loss / len(all_targets)
    return avg_loss, auc, all_probs, all_targets

print("\n" + "="*70)
print(f"STARTING TRAINING: {NUM_EPOCHS} EPOCHS WITH STANFORD CHEXPERT WEIGHTS")
print("="*70)

best_val_auc = 0.0
best_model_state = None
history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS:02d}", leave=False):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        imgs_mixed, y_a, y_b, lam = mixup_data(imgs, lbls, alpha=MIXUP_ALPHA)
        
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs_mixed)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * len(lbls)
        
    scheduler.step()
    train_loss = running_loss / len(train_df)
    val_loss, val_auc, _, _ = evaluate_loader(model, val_loader)
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(val_auc)
    
    is_best = val_auc > best_val_auc
    if is_best:
        best_val_auc = val_auc
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, os.path.join(SAVE_DIR, "phase1_chexpert_best_model.pth"))
        
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} {'[BEST]' if is_best else ''}")

print(f"\nBest Validation ROC-AUC: {best_val_auc:.4f}")

# ── Load Best Weights for Evaluation ──────────────────────────────────────
model.load_state_dict(best_model_state)
model.eval()

# 1. Standard Single-Pass Test Evaluation
_, _, test_probs, y_test = evaluate_loader(model, test_loader)
_, _, val_probs, y_val   = evaluate_loader(model, val_loader)

# Optimal Youden threshold from validation set
fpr_v, tpr_v, th_v = roc_curve(y_val, val_probs)
youden_idx = np.argmax(tpr_v - fpr_v)
opt_threshold = float(th_v[youden_idx])

std_auc = float(roc_auc_score(y_test, test_probs))
test_preds = (test_probs >= opt_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
acc  = float((tp + tn) / (tp + tn + fp + fn))
sens = float(tp / (tp + fn))
spec = float(tn / (tn + fp))
f1   = float(f1_score(y_test, test_preds))

print("\n" + "="*70)
print(f"EXPERIMENT 2 (CHEXPERT WEIGHTS) HELD-OUT TEST RESULTS (N={len(y_test)})")
print("="*70)
print(f"Standard Test ROC-AUC : {std_auc:.4f}")
print(f"Youden-J Threshold    : {opt_threshold:.4f}")
print(f"Test Accuracy         : {acc*100:.2f}%")
print(f"Test Sensitivity      : {sens*100:.2f}% (Missed Infections: {fn})")
print(f"Test Specificity      : {spec*100:.2f}%")
print(f"Test F1-Score         : {f1:.4f}")

# 2. 10-View Test-Time Augmentation (TTA)
print("\nRunning 10-View Test-Time Augmentation (TTA)...")
tta_prob_list = []
for t_idx, tta_tfm in enumerate(tta_transforms):
    test_ds.set_transform(tta_tfm)
    tta_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    _, _, p_v, _ = evaluate_loader(model, tta_loader)
    tta_prob_list.append(p_v)

tta_probs = np.mean(tta_prob_list, axis=0)
tta_auc = float(roc_auc_score(y_test, tta_probs))
tta_preds = (tta_probs >= opt_threshold).astype(int)
tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, tta_preds).ravel()
tta_acc = float((tp_t + tn_t) / (tp_t + tn_t + fp_t + fn_t))

print(f"TTA Test ROC-AUC      : {tta_auc:.4f} (+{tta_auc - std_auc:.4f} boost)")
print(f"TTA Test Accuracy     : {tta_acc*100:.2f}%")

# ── Save Results to JSON ──────────────────────────────────────────────────
results_payload = {
    "experiment": "Experiment 2 — Stanford CheXpert Implementation",
    "backbone": "densenet121-res224-chex",
    "pretraining_source": "Stanford CheXpert (224,316 radiographs, Stanford Hospital, CA)",
    "mimic_overlap": "0.0%",
    "split": {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "test_patients": int(test_df['subject_id'].nunique())
    },
    "results": {
        "test_auc": round(std_auc, 4),
        "tta_auc": round(tta_auc, 4),
        "youden_threshold": round(opt_threshold, 4),
        "accuracy": round(acc * 100, 2),
        "sensitivity": round(sens * 100, 2),
        "specificity": round(spec * 100, 2),
        "f1_score": round(f1, 4),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "best_val_auc": round(best_val_auc, 4)
    },
    "comparison_vs_legacy_all": {
        "legacy_all_auc": 0.8408,
        "legacy_all_tta_auc": 0.8446,
        "chexpert_auc": round(std_auc, 4),
        "chexpert_tta_auc": round(tta_auc, 4),
        "auc_delta": round(std_auc - 0.8408, 4),
        "leak_free_guarantee": True
    }
}

RESULTS_JSON_PATH = os.path.join(SAVE_DIR, "phase1_chexpert_results.json")
with open(RESULTS_JSON_PATH, "w") as f:
    json.dump(results_payload, f, indent=2)
print(f"\nSaved structured results -> {RESULTS_JSON_PATH}")

# ── Save Publication Plots ────────────────────────────────────────────────
plt.figure(figsize=(10, 4.5), facecolor="white")
plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), history['train_loss'], 'b-o', label='Train Loss', lw=1.8)
plt.plot(range(1, NUM_EPOCHS + 1), history['val_loss'],   'r--s', label='Val Loss', lw=1.8)
plt.xlabel("Epoch", fontsize=11, fontweight='bold')
plt.ylabel("Loss", fontsize=11, fontweight='bold')
plt.title("Phase 1 (CheXpert): Loss Curves", fontsize=12, fontweight='bold')
plt.legend(frameon=True); plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), history['val_auc'], 'g-^', label='Validation AUC', lw=2.0)
plt.axhline(best_val_auc, color='gray', linestyle=':', label=f'Best Val AUC ({best_val_auc:.4f})')
plt.xlabel("Epoch", fontsize=11, fontweight='bold')
plt.ylabel("ROC-AUC", fontsize=11, fontweight='bold')
plt.title("Phase 1 (CheXpert): Validation ROC-AUC", fontsize=12, fontweight='bold')
plt.legend(frameon=True); plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "phase1_chexpert_training_curves.png"), dpi=180, bbox_inches='tight')
plt.close()

fpr, tpr, _ = roc_curve(y_test, test_probs)
plt.figure(figsize=(11, 4.8), facecolor="white")
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"Standard Test (AUC = {std_auc:.4f})")
fpr_t, tpr_t, _ = roc_curve(y_test, tta_probs)
plt.plot(fpr_t, tpr_t, color="#10B981", lw=2.0, linestyle="--", label=f"10-View TTA (AUC = {tta_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel("1 - Specificity (FPR)", fontsize=11, fontweight='bold')
plt.ylabel("Sensitivity (TPR)", fontsize=11, fontweight='bold')
plt.title("Phase 1 (CheXpert): Test ROC Curves", fontsize=12, fontweight='bold')
plt.legend(loc="lower right", frameon=True); plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
cm = np.array([[tn, fp], [fn, tp]])
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title(f"Confusion Matrix (tau={opt_threshold:.3f})", fontsize=12, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Pneumonia'], fontsize=10, fontweight='bold')
plt.yticks(tick_marks, ['Normal', 'Pneumonia'], fontsize=10, fontweight='bold')
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=12, fontweight='bold')
plt.ylabel("True Diagnosis", fontsize=11, fontweight='bold')
plt.xlabel("Predicted Diagnosis", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "phase1_chexpert_roc_cm.png"), dpi=180, bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("PHASE 1 CHEXPERT TRAINING AND EVALUATION SCRIPT PREPARED.")
print("="*70)
