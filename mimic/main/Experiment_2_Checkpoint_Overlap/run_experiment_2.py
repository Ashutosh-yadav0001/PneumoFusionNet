"""
mimic/main/Experiment_2_Checkpoint_Overlap/run_experiment_2.py
============================================================
Experiment 2 — Resolve Image Checkpoint Overlap
PneumoFusionNet | MIMIC-CXR PA Scale-up Cohort (N = 3,763)

Investigates whether the pretraining mixture of 'densenet121-res224-all' (which
includes MIMIC-CXR) causes data leakage or artificial inflation, and benchmarks
alternative pretrained checkpoints with documented training data strictly
excluding MIMIC-CXR:

1. 'densenet121-res224-all'  : TorchXRayVision multi-dataset mixture (contains MIMIC-CXR)
2. 'densenet121-res224-nih'  : NIH ChestX-ray14 only (112,120 frontal CXRs, 0% MIMIC overlap)
3. 'densenet121-res224-chex' : Stanford CheXpert only (224,316 CXRs, 0% MIMIC overlap)
4. 'imagenet'                : TorchVision ImageNet-1K (1.28M natural RGB photos, 0% MIMIC overlap)

Evaluates on the exact canonical Experiment 1 test partition (N_test = 562, 511 patients).
"""

import os
import sys
import json
import random
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as tv_models
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# ── Reproducibility & Device ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = r"C:\2026\PneumoFusionNet\mimic\main"
CSV_PATH    = os.path.join(BASE_DIR, "dataset", "mimic_paired_dataset.csv")
SPLIT_CSV   = os.path.join(BASE_DIR, "dataset", "patient_split.csv")
BBOX_CSV    = os.path.join(BASE_DIR, "outputs", "Phase_1.1v4_PA_crossval_scaleup", "lung_bboxes.csv")
OUT_DIR     = os.path.join(BASE_DIR, "outputs", "Experiment_2_checkpoint_overlap")
os.makedirs(OUT_DIR, exist_ok=True)

# ── DeLong Test Implementation for Paired ROC-AUC Comparison ──────────────
def delong_roc_variance(ground_truth, predictions):
    """Computes variance of AUC using DeLong's algorithm."""
    order = np.lexsort((predictions, ground_truth))
    ground_truth = ground_truth[order]
    predictions = predictions[order]
    
    n_pos = np.sum(ground_truth == 1)
    n_neg = np.sum(ground_truth == 0)
    
    pos_preds = predictions[ground_truth == 1]
    neg_preds = predictions[ground_truth == 0]
    
    # Structural components
    v10 = np.array([np.mean(pos > neg_preds) + 0.5 * np.mean(pos == neg_preds) for pos in pos_preds])
    v01 = np.array([np.mean(neg < pos_preds) + 0.5 * np.mean(neg == pos_preds) for neg in neg_preds])
    
    var_v10 = np.var(v10, ddof=1) if n_pos > 1 else 0
    var_v01 = np.var(v01, ddof=1) if n_neg > 1 else 0
    
    auc = np.mean(v10)
    var_auc = (var_v10 / n_pos) + (var_v01 / n_neg)
    return auc, var_auc, v10, v01

def delong_paired_pvalue(y_true, y_prob1, y_prob2):
    """Computes paired DeLong p-value between two models on identical ground truth."""
    auc1, var1, v10_1, v01_1 = delong_roc_variance(y_true, y_prob1)
    auc2, var2, v10_2, v01_2 = delong_roc_variance(y_true, y_prob2)
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    cov_v10 = np.cov(v10_1, v10_2)[0, 1] if n_pos > 1 else 0
    cov_v01 = np.cov(v01_1, v01_2)[0, 1] if n_neg > 1 else 0
    cov = (cov_v10 / n_pos) + (cov_v01 / n_neg)
    
    var_diff = var1 + var2 - 2 * cov
    if var_diff <= 0:
        return 1.0, auc1, auc2
    
    z = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value, auc1, auc2

# ── Load Dataset & Shared Patient Split ────────────────────────────────────
print("\n" + "="*70)
print("Loading Cohort and Canonical Patient Split...")
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

# ── Dataset Definition ────────────────────────────────────────────────────
class CXRExp2Dataset(Dataset):
    def __init__(self, data_df, bbox_dict, is_imagenet=False):
        self.df = data_df
        self.bbox_dict = bbox_dict
        self.is_imagenet = is_imagenet
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if is_imagenet:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = int(row['label'])
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)
        else:
            img = self.clahe.apply(img)
            bb = self.bbox_dict.get(img_path, None)
            if bb and bb.get('x_max', 0) > 0 and (bb['x_max'] - bb['x_min'] > 30) and (bb['y_max'] - bb['y_min'] > 30):
                img = img[bb['y_min']:bb['y_max'], bb['x_min']:bb['x_max']]
        
        pil_img = Image.fromarray(img)
        t = self.transform(pil_img)
        
        if not self.is_imagenet:
            # TorchXRayVision standard scaling: [-1024, 1024]
            t = (t * 2048.0) - 1024.0
            
        return t, label

# ── Checkpoints to Investigate ────────────────────────────────────────────
CHECKPOINTS = [
    {
        "id": "all",
        "name": "DenseNet-121 (All CXR Mixture)",
        "weights": "densenet121-res224-all",
        "training_data": "NIH, PadChest, CheXpert, MIMIC-CXR, OpenI, Google, RSNA",
        "mimic_overlap": "YES (Trained on MIMIC-CXR CheXpert set)",
        "is_imagenet": False,
        "color": "#DC2626"
    },
    {
        "id": "nih",
        "name": "DenseNet-121 (NIH ChestX-ray14 Only)",
        "weights": "densenet121-res224-nih",
        "training_data": "NIH ChestX-ray14 only (112,120 frontal CXRs, 30,805 patients)",
        "mimic_overlap": "NO (0.0% overlap; strictly Bethesda, MD cohort)",
        "is_imagenet": False,
        "color": "#2563EB"
    },
    {
        "id": "chex",
        "name": "DenseNet-121 (Stanford CheXpert Only)",
        "weights": "densenet121-res224-chex",
        "training_data": "Stanford CheXpert only (224,316 radiographs, 65,240 patients)",
        "mimic_overlap": "NO (0.0% overlap; strictly Stanford Hospital cohort)",
        "is_imagenet": False,
        "color": "#16A34A"
    },
    {
        "id": "imagenet",
        "name": "DenseNet-121 (TorchVision ImageNet-1K)",
        "weights": "imagenet",
        "training_data": "ImageNet-1K (1.28 million natural photographic RGB images)",
        "mimic_overlap": "NO (0.0% overlap; natural consumer images)",
        "is_imagenet": True,
        "color": "#9333EA"
    }
]

# ── Helper: Build Model Backbone ──────────────────────────────────────────
def get_backbone(ckpt_cfg):
    w = ckpt_cfg["weights"]
    if ckpt_cfg["is_imagenet"]:
        tv_model = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.DEFAULT)
        old_conv = tv_model.features.conv0
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        tv_model.features.conv0 = new_conv
        model = tv_model
        features = tv_model.features
    else:
        xrv_model = xrv.models.DenseNet(weights=w)
        model = xrv_model
        features = xrv_model.features
    
    features = features.to(DEVICE)
    features.eval()
    return model, features

# ── Extract Features Function ─────────────────────────────────────────────
def extract_features(loader, features_net, pool_layer):
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, leave=False):
            imgs = imgs.to(DEVICE)
            f_map = F.relu(features_net(imgs), inplace=True)
            pooled = pool_layer(f_map).flatten(1).cpu().numpy()
            all_feats.append(pooled)
            all_labels.append(lbls.numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)

# ── Main Experiment Execution ─────────────────────────────────────────────
print("\n" + "="*70)
print("EXECUTING EXPERIMENT 2: CHECKPOINT OVERLAP AUDIT & EVALUATION")
print("="*70)

pool_layer = nn.AdaptiveAvgPool2d(1).to(DEVICE)
results_summary = []
y_prob_dict = {}
y_true_test = None

for cfg in CHECKPOINTS:
    cid   = cfg["id"]
    cname = cfg["name"]
    print(f"\n--- Testing Checkpoint: {cname} ---")
    print(f"Training Data : {cfg['training_data']}")
    print(f"MIMIC Overlap : {cfg['mimic_overlap']}")
    
    # 1. Instantiate backbone
    raw_model, feat_extractor = get_backbone(cfg)
    
    # 2. Build datasets and loaders
    train_ds = CXRExp2Dataset(train_df, bbox_lookup, is_imagenet=cfg["is_imagenet"])
    val_ds   = CXRExp2Dataset(val_df, bbox_lookup, is_imagenet=cfg["is_imagenet"])
    test_ds  = CXRExp2Dataset(test_df, bbox_lookup, is_imagenet=cfg["is_imagenet"])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Extract 1024-dim features
    print("Extracting features across splits...")
    X_train, y_train = extract_features(train_loader, feat_extractor, pool_layer)
    X_val,   y_val   = extract_features(val_loader, feat_extractor, pool_layer)
    X_test,  y_test  = extract_features(test_loader, feat_extractor, pool_layer)
    
    if y_true_test is None:
        y_true_test = y_test
    
    # 4. Dimension A: Zero-Shot Pneumonia Pathology Head (if medical CXR model)
    zero_shot_auc = None
    if not cfg["is_imagenet"] and hasattr(raw_model, "pathologies") and "Pneumonia" in raw_model.pathologies:
        pn_idx = raw_model.pathologies.index("Pneumonia")
        print(f"Evaluating zero-shot pneumonia prediction (pathology index {pn_idx})...")
        raw_model = raw_model.to(DEVICE).eval()
        zs_probs = []
        with torch.no_grad():
            for imgs, _ in tqdm(test_loader, leave=False):
                imgs = imgs.to(DEVICE)
                preds = raw_model(imgs)
                # sigmoid activation if not applied
                prob = torch.sigmoid(preds[:, pn_idx]).cpu().numpy()
                zs_probs.append(prob)
        zs_probs = np.concatenate(zs_probs)
        zero_shot_auc = float(roc_auc_score(y_test, zs_probs))
        print(f"Zero-shot Pretrained AUC on MIMIC-CXR test set: {zero_shot_auc:.4f}")
    
    # 5. Dimension B: Linear Probe Classifier on Train Split, Tested on Identical Test Split
    print("Fitting linear diagnostic probe on train features...")
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]
    
    # Optimize Youden's threshold on validation set
    fpr_val, tpr_val, th_val = roc_curve(y_val, val_probs)
    youden_idx = np.argmax(tpr_val - fpr_val)
    opt_threshold = float(th_val[youden_idx])
    
    # Evaluate on held-out test set
    test_auc = float(roc_auc_score(y_test, test_probs))
    test_preds = (test_probs >= opt_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    acc = float((tp + tn) / (tp + tn + fp + fn))
    sens = float(tp / (tp + fn))
    spec = float(tn / (tn + fp))
    
    y_prob_dict[cid] = test_probs
    
    res = {
        "id": cid,
        "name": cname,
        "training_data": cfg["training_data"],
        "mimic_overlap": cfg["mimic_overlap"],
        "zero_shot_auc": zero_shot_auc,
        "linear_probe_auc": round(test_auc, 4),
        "youden_threshold": round(opt_threshold, 4),
        "accuracy": round(acc * 100, 2),
        "sensitivity": round(sens * 100, 2),
        "specificity": round(spec * 100, 2),
        "test_n": len(y_test)
    }
    results_summary.append(res)
    print(f"RESULTS -> Test AUC: {test_auc:.4f} | Acc: {acc*100:.2f}% | Sens: {sens*100:.2f}% | Spec: {spec*100:.2f}%")

# ── DeLong Paired Statistical Tests vs. Baseline ('all') ──────────────────
print("\n" + "="*70)
print("DELONG STATISTICAL SIGNIFICANCE TESTING (vs. 'densenet121-res224-all')")
print("="*70)

baseline_probs = y_prob_dict["all"]
for res in results_summary:
    cid = res["id"]
    if cid == "all":
        res["delong_p_value_vs_all"] = 1.0
        res["delong_significance"] = "Baseline"
    else:
        comp_probs = y_prob_dict[cid]
        p_val, auc_base, auc_comp = delong_paired_pvalue(y_true_test, baseline_probs, comp_probs)
        res["delong_p_value_vs_all"] = round(float(p_val), 5)
        res["delong_significance"] = "Statistically Significant (p < 0.05)" if p_val < 0.05 else "No Significant Difference (p >= 0.05)"
        print(f"Comparison: 'all' (AUC={auc_base:.4f}) vs '{cid}' (AUC={auc_comp:.4f}) -> p-value: {p_val:.5f} [{res['delong_significance']}]")

# ── Save Results to JSON ──────────────────────────────────────────────────
RESULTS_FILE = os.path.join(OUT_DIR, "experiment_2_results.json")
with open(RESULTS_FILE, 'w') as f:
    json.dump({
        "experiment": "Experiment 2 — Resolve Image Checkpoint Overlap",
        "description": "Systematic audit and benchmark of alternative image backbones strictly excluding MIMIC-CXR",
        "test_size": len(y_true_test),
        "results": results_summary
    }, f, indent=2)
print(f"\nSaved structured results -> {RESULTS_FILE}")

# ── Publication-Quality Figures ───────────────────────────────────────────
print("\nGenerating publication-quality comparison figures...")

# 1. ROC Curves
plt.figure(figsize=(9, 8), facecolor="white")
ax = plt.subplot(1, 1, 1)
ax.set_facecolor("white")

for cfg in CHECKPOINTS:
    cid = cfg["id"]
    probs = y_prob_dict[cid]
    fpr, tpr, _ = roc_curve(y_true_test, probs)
    auc_val = roc_auc_score(y_true_test, probs)
    
    is_base = (cid == "all")
    label_txt = f"{cfg['name']} (AUC = {auc_val:.4f})"
    if is_base:
        label_txt += " [Contains MIMIC]"
    else:
        label_txt += " [Zero Leakage]"
        
    ax.plot(fpr, tpr, color=cfg["color"], lw=2.5 if is_base else 2.0,
            linestyle="--" if is_base else "-", label=label_txt)

ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.5000)")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.02])
ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=13, fontweight='bold', labelpad=8)
ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=13, fontweight='bold', labelpad=8)
ax.set_title("Experiment 2: ROC Comparison of Alternative Image Backbones\n(Canonical Experiment 1 Test Cohort: N = 562, 511 Patients)",
             fontsize=14, fontweight='bold', pad=15)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right", fontsize=11, frameon=True, edgecolor="#333333")
plt.tight_layout()

ROC_FIG_PATH = os.path.join(OUT_DIR, "exp2_roc_curves.png")
plt.savefig(ROC_FIG_PATH, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved ROC plot -> {ROC_FIG_PATH}")

# 2. Bar Chart Comparison
fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="white")
ax.set_facecolor("white")

labels = [c["id"].upper() for c in CHECKPOINTS]
aucs = [r["linear_probe_auc"] for r in results_summary]
sens = [r["sensitivity"] / 100.0 for r in results_summary]
specs = [r["specificity"] / 100.0 for r in results_summary]
accs = [r["accuracy"] / 100.0 for r in results_summary]

x = np.arange(len(labels))
w = 0.20

b1 = ax.bar(x - 1.5*w, aucs, w, label="ROC-AUC", color="#2563EB", edgecolor="black")
b2 = ax.bar(x - 0.5*w, accs, w, label="Accuracy", color="#10B981", edgecolor="black")
b3 = ax.bar(x + 0.5*w, sens, w, label="Sensitivity", color="#F59E0B", edgecolor="black")
b4 = ax.bar(x + 1.5*w, specs, w, label="Specificity", color="#8B5CF6", edgecolor="black")

for bars in [b1, b2, b3, b4]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([
    "ALL (CXR Mixture)\n[Contains MIMIC]",
    "NIH ChestX-ray14\n[0% MIMIC Overlap]",
    "Stanford CheXpert\n[0% MIMIC Overlap]",
    "ImageNet-1K\n[0% MIMIC Overlap]"
], fontsize=11, fontweight='bold')
ax.set_ylim(0.50, 1.08)
ax.set_ylabel("Score", fontsize=12, fontweight='bold')
ax.set_title("Experiment 2: Metric Comparison Across Pretrained Backbones\nEvaluating Impact of Image Checkpoint Pretraining on Held-out MIMIC-CXR Test Cohort",
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc="lower right", frameon=True, edgecolor="#333333")
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

BAR_FIG_PATH = os.path.join(OUT_DIR, "exp2_metrics_bar.png")
plt.savefig(BAR_FIG_PATH, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved Bar plot -> {BAR_FIG_PATH}")

print("\n" + "="*70)
print("EXPERIMENT 2 COMPLETE.")
print("="*70)
