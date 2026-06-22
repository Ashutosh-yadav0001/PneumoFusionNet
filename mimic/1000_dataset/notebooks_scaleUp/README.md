# 📓 Phase 1.1 — Scale-Up Notebooks (PA / AP View Classifiers)

> **Project**: PneumoFusionNet — Pneumonia Detection from MIMIC-CXR Chest X-rays  
> **Task**: Binary Classification → `NORMAL` vs `PNEUMONIA`  
> **Dataset Source**: MIMIC-CXR-JPG (PhysioNet) — 2,221 paired image–report subset

---

## 📂 Notebook Index

| # | Notebook | View | What It Does |
|:-:|----------|:----:|-------------|
| 1 | `Phase-1.1-image_classifier_balanced_scaleUp(PA View).ipynb` | PA | Baseline balanced classifier with standard augmentations and BCE loss |
| 2 | `Phase-1.1-image_classifier_balanced_scaleUp(PA View 500+) - Copy.ipynb` | PA | Experimental copy of Notebook 1 for parameter tuning |
| 3 | `Phase-1.1v2-enhanced_image_classifier_PA.ipynb` | PA | Enhanced: lung-ROI crop (torchxrayvision), CLAHE, deeper fine-tuning |
| 4 | `Phase-1.1v3-fixed_image_classifier_PA.ipynb` | PA | Mixup (α=0.3), label smoothing, cosine LR, class-weighted loss, threshold tuning |
| 5 | `Phase-1.1v4-crossval_tta_PA.ipynb` | PA | 5-fold stratified CV + 10-view TTA + per-fold threshold tuning |

---

## 📊 Dataset & Sample Counts

| Version | View | Train | Val | Test | Total | Normal : Pneumonia | Balance |
|:-------:|:----:|:-----:|:---:|:----:|:-----:|:------------------:|:-------:|
| **v1 PA** | PA | 590 | 130 | 134 | **854** | 318 : 272 (train) | ~54:46 |
| **v1 AP** | AP | — | — | 64 | **~200** | 31 : 33 (test) | ~48:52 |
| **v2** | PA | 246 | 55 | 53 | **354** | 123 : 123 (train) | 50:50 |
| **v3** | PA | 590 | 130 | 134 | **854** | 318 : 272 (train) | ~54:46 |
| **v4** | PA | 854 total across 5 folds | — | — | **854** | Same as v1/v3 | ~54:46 |

> ⚠️ **v2 used only 354 samples** (strict PA + lung-crop filtering removed ~500 images) — this severely limited training.

---

## 🏗️ Model Architecture

| Version | Backbone | Pretrained | Unfrozen Layers | Classifier Head | Total Params |
|:-------:|----------|:----------:|----------------|----------------|:------------:|
| **v1** | DenseNet-121 | ImageNet | Last dense block | Linear(1024→2) | ~7.0M |
| **v2** | DenseNet-121 | ImageNet | Last 2 dense blocks | Linear(1024→2) | ~7.0M |
| **v3** | DenseNet-121 | ImageNet | Last 2 dense blocks | Linear(1024→2) | ~7.0M |
| **v4** | DenseNet-121 | ImageNet | Last 2 dense blocks | Linear(1024→2) | ~7.0M |

| Version | Loss Function | Optimizer | LR | Scheduler | Epochs | Batch |
|:-------:|:-------------|:---------:|:--:|-----------|:------:|:-----:|
| **v1** | BCE | Adam | 1e-4 | None | 25 | 16 |
| **v2** | BCE | Adam | 1e-4 | StepLR | 20 | 16 |
| **v3** | BCE + Label Smooth (0.1) + Weighted (2× PN) | AdamW | 3e-5 | Cosine + Warmup (5ep) | 50 | 8 |
| **v4** | BCE + Label Smooth (0.1) + Weighted (2× PN) | AdamW | 3e-5 | Cosine + Warmup (5ep) | 25 | 8 |

---

## 📈 Accuracy & Performance Results

### Master Comparison Table

| Version | Samples | AUC | Accuracy | Sensitivity | Specificity | F1 | AP |
|:-------:|:-------:|:---:|:--------:|:-----------:|:-----------:|:--:|:--:|
| **v1 PA** | 854 | **0.8348** | 69.4% | 65.3% | 74.6% | 0.69 | — |
| **v1 AP** | ~200 | 0.6158 | 57.8% | 84.8% | 29.0% | 0.53 | — |
| **v2 Enhanced** | 354 | 0.7251 | 58.5% | 30.8% | 85.2% | 0.51 | — |
| **v3 Fixed** | 854 | 0.7137 | 64.2% | 65.4% | 63.0% | 0.64 | 0.7418 |
| **v3 Tuned** (θ=0.535) | 854 | 0.7137 | 66.0% | 42.3% | 88.9% | 0.60 | 0.7418 |
| **v3 Val-best** | 854 | 0.8197 | 70.9% | — | — | 0.71 | — |
| **v4 CV Pooled** | 854 | 0.7490 | 69.2% | 68.9% | 69.5% | 0.69 | — |
| **v4 TTA Pooled** | 854 | **0.7613** | — | — | — | — | — |
| **v4 Tuned** (θ=0.515) | 854 | — | 70.8% | 64.4% | 76.8% | 0.71 | — |

### v4 — Per-Fold Detailed Results

| Fold | Best Epoch | Val AUC | Test AUC | Test Acc | TTA AUC | Tuned Acc | PN Recall | PN Specificity |
|:----:|:---------:|:-------:|:--------:|:--------:|:-------:|:---------:|:---------:|:--------------:|
| 1 | 9 | 0.750 | 0.817 | 78.9% | 0.844 | 81.7% | 77.8% | 85.7% |
| 2 | 9 | 0.834 | 0.716 | 67.6% | 0.723 | 70.4% | 81.3% | 61.5% |
| 3 | 20 | 0.667 | 0.750 | 63.4% | 0.763 | 76.1% | 59.4% | 89.7% |
| 4 | 12 | 0.773 | 0.800 | 70.4% | 0.806 | 78.9% | 81.1% | 76.5% |
| 5 | 5 | 0.800 | 0.740 | 67.1% | 0.729 | 71.4% | 57.5% | 90.0% |
| **Mean** | **11** | **0.765** | **0.765** | **69.5%** | **0.773** | **75.7%** | **71.4%** | **80.7%** |
| **Std** | **±5.7** | **±0.06** | **±0.04** | **±5.7%** | **±0.05** | **±4.8%** | **±11.9%** | **±11.6%** |

---

## 🔴 Identified Issues

### Issue #1: AUC Dropped from 0.83 → 0.71 (v1 → v2/v3)

| Issue | What Went Wrong | Version | Severity |
|:-----:|----------------|:-------:|:--------:|
| Aggressive lung-ROI crop | torchxrayvision bboxes cut out peri-hilar and lower-lobe pneumonia regions | v2 | 🔴 High |
| CLAHE over-enhancement | Amplified noise/edge artifacts in noisy MIMIC images; confused the model | v2 | 🔴 High |
| Too few samples after filtering | v2 training used only 246 samples (vs 590 in v1) — 58% data loss | v2 | 🔴 High |
| Mixup too strong (α=0.3) | Blending images dilutes already scarce pneumonia patterns on small dataset | v3 | 🟡 Medium |
| Label smoothing too high (0.1) | Targets become [0.05, 0.95] — prevents confident learning on 2-class task | v3 | 🟡 Medium |

### Issue #2: Training Instability

| Issue | Evidence | Version | Severity |
|:-----:|----------|:-------:|:--------:|
| Overfitting | Val loss diverges upward from epoch 5–10 while train loss drops steadily | v1, v3 | 🔴 High |
| Val accuracy oscillation | ±10% accuracy swings between consecutive epochs | v3 | 🟡 Medium |
| Fold variance too high | PN recall ranges from 57% (Fold 5) to 81% (Fold 4) — Δ = 24% | v4 | 🟡 Medium |
| Convergence speed varies | Best epoch ranges from 5 to 20 across folds | v4 | 🟡 Medium |

### Issue #3: Clinical Viability

| Issue | Current | Target | Gap | Severity |
|:-----:|:-------:|:------:|:---:|:--------:|
| Low sensitivity (PN recall) | 65.4% (v3) / 71.4% (v4 mean) | ≥ 80% | **-9 to -15%** | 🔴 Critical |
| Poor calibration | θ shift 0.50→0.54 swings sensitivity 65%→42% | Stable across thresholds | **Uncalibrated** | 🔴 High |
| AP-view near random | AUC = 0.62 | ≥ 0.80 | **-0.18** | 🟡 Medium |
| AUC below clinical bar | 0.76 (pooled CV) | ≥ 0.85 | **-0.09** | 🔴 High |

---

## 🚀 Up Next — Planned Improvements

### 🔥 Immediate (Next Iteration)

| # | What | Why | Expected Gain |
|:-:|------|-----|:-------------:|
| 1 | **Scale dataset to 2,000–5,000 PA samples** | 854 samples is too small for 7M-param DenseNet; high fold variance confirms this | AUC +0.05–0.10 |
| 2 | **Remove Mixup & reduce label smoothing to 0.02** | Both hurt performance on small medical datasets — v1 without them scored higher | AUC +0.03–0.05 |
| 3 | **Revert to simple preprocessing** | Drop CLAHE + lung-ROI crop; use histogram EQ + center-crop instead | Recover to v1 AUC |
| 4 | **Switch to Focal Loss (γ=2.0)** | Handles hard examples better than static 2× class weight | Sensitivity +5–10% |
| 5 | **Add medical-specific augmentations** | Elastic deform, random gamma, Gaussian noise — better than Mixup for CXR | Generalization ↑ |

### ⚡ Short-Term (v5 Architecture)

| # | What | Why | Expected Gain |
|:-:|------|-----|:-------------:|
| 6 | **Try EfficientNet-B3 or ConvNeXt-Tiny** | More parameter-efficient; better feature extraction on limited data | AUC +0.02–0.05 |
| 7 | **Progressive unfreezing (1 block/epoch)** | Prevents catastrophic forgetting of pretrained features | Stability ↑ |
| 8 | **CutMix instead of Mixup** | Preserves spatial features (consolidation regions) unlike pixel-blending | Localization ↑ |
| 9 | **Ensemble top folds (1, 2, 4)** | Average predictions from best 3 folds to reduce variance | AUC +0.01–0.03 |
| 10 | **Add Platt scaling / temperature calibration** | Fix threshold sensitivity — model outputs are not calibrated | Calibration ↑ |

### 🔮 Future Work (Phase 2+)

| # | What | Why |
|:-:|------|-----|
| 11 | **Multi-view fusion (PA + AP)** | Per-patient prediction combining both views |
| 12 | **GradCAM / Attention visualization** | Verify model focuses on lung parenchyma, not artifacts |
| 13 | **External validation (CheXpert / NIH)** | Measure cross-dataset generalization |
| 14 | **Report-guided multimodal training** | Use radiology text as auxiliary supervision (Phase 2 fusion) |

---

## 📁 Output Directories

| Directory | Model Size | Key Files |
|-----------|:----------:|-----------|
| `outputs/Phase_1.1_PA_balanced_scaled/` | 100 MB | Model, confusion matrix, ROC, training history, extracted features (train/val/test) |
| `outputs/Phase_1.1_AP_balanced_scaled/` | 100 MB | Model, confusion matrix, ROC, training history |
| `outputs/Phase_1.1v2_PA_enhanced/` | 30 MB | Model, lung bboxes CSV, preprocessing comparison, confusion matrix, ROC |
| `outputs/Phase_1.1v3_PA_fixed/` | 30 MB | Model, confusion matrices (2 thresholds), ROC/PR curves, threshold plot, config |
| `outputs/Phase_1.1v4_PA_crossval/` | 150 MB | 5 fold checkpoints, CV results CSV, pooled confusion matrices, pooled ROC, config |

---

## ⚙️ Environment

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 1.12 |
| torchvision | ≥ 0.13 |
| torchxrayvision | ≥ 1.4.0 |
| albumentations | ≥ 1.3.1 |
| scikit-learn | ≥ 1.3.2 |
| GPU | NVIDIA CUDA-enabled (required) |

See [requirements.txt](../../../requirements.txt) for full dependency list.

---

## 📌 Current Status

| Metric | Best Value | Source | Target |
|--------|:----------:|:------:|:------:|
| **AUC (single split)** | 0.8348 | v1 PA | ≥ 0.85 |
| **AUC (cross-validated)** | 0.7613 | v4 TTA pooled | ≥ 0.85 |
| **Sensitivity** | 71.4% | v4 mean | ≥ 80% |
| **Specificity** | 80.7% | v4 mean | ≥ 75% |
| **Accuracy** | 75.7% | v4 tuned mean | ≥ 80% |

> **Bottom line**: The model is not yet clinically viable. The primary bottleneck is **insufficient data** (854 samples) causing high variance and overfitting. Scaling to 2,000+ samples with simpler preprocessing is the highest-priority next step.
