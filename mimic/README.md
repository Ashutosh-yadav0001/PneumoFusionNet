<div align="center">

# 🫁 PneumoFusionNet

### Triple-Fusion Multimodal AI for Pneumonia Detection
**Image · Radiology Text · Clinical Metadata**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-MIMIC--CXR-green?style=flat-square)](https://physionet.org/content/mimic-cxr/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 🏆 Key Results

| Model | Dataset | AUC | Sensitivity | Specificity | Accuracy |
|:---|:---:|:---:|:---:|:---:|:---:|
| Phase 1 — Image Only | ~1,857 | 0.8152 | 69.7% | — | 76.0% |
| Phase 1 — Image Only | ~3,763 | 0.8258 | 71.0% | — | 76.3% |
| Phase 2 — Image + Text | ~1,857 | 0.9490 | 91.4% | 86.3% | 88.6% |
| Phase 2 — Image + Text | ~3,763 | 0.9460 | 90.3% | 89.1% | 87.8% |
| Phase 3 — Triple Fusion | ~1,857 | 0.9841 | **94.9%** | **94.5%** | **94.7%** |
| **Phase 3 — Triple Fusion** | **~3,763** | **0.9890** | 89.1% | 94.3% | 91.7% |

> 🎯 **Best AUC: 0.9890** on 3,763-image scaleup dataset · Phase 3 Triple Fusion

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline Phases](#pipeline-phases)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Clinical Notes](#clinical-notes)
- [Citation](#citation)

---

## Overview

**PneumoFusionNet** is a three-phase multimodal deep learning pipeline for pneumonia detection using chest X-rays from the MIMIC-CXR dataset. The project progressively fuses three clinical modalities:

```
Phase 1 ──► Image Only           (DenseNet-121 + CBAM)
Phase 2 ──► Image + Text         (+ Bio_ClinicalBERT + CrossAttention)
Phase 3 ──► Image + Text + Meta  (+ Clinical Metadata MLP)
```

Each phase demonstrates measurable improvement, providing a clean ablation study of multimodal fusion in medical imaging.

---

## Architecture

### Phase 3 — Triple Fusion Network

```
┌─────────────────────────────────────────────────────────────────┐
│                     PNEUMOFUSIONNET (Phase 3)                   │
├─────────────────┬──────────────────────┬────────────────────────┤
│   IMAGE BRANCH  │    TEXT BRANCH       │   METADATA BRANCH      │
│                 │                      │                        │
│  Chest X-Ray    │  Radiology Report    │  Clinical Labs/Vitals  │
│      │          │       │              │         │              │
│  DenseNet-121   │  Bio_ClinicalBERT    │   MLP (17→128→64)      │
│   + CBAM Attn   │  (last 2 unfrozen)   │                        │
│      │          │       │              │         │              │
│  1024-d feat    │  768-d tokens        │    64-d embedding      │
│      │          │       │              │                        │
│      └──────────┤  CrossAttention      │                        │
│                 │  (img queries text)  │                        │
│                 │       │              │                        │
│              512-d attended feat       │                        │
│                 │                      │                        │
└────────────────[concat: 1024 + 512 + 64 = 1600-d]──────────────┘
                              │
                    LayerNorm → Dropout
                    Linear(1600→512) → GELU
                    Linear(512→128)  → GELU
                    Linear(128→2)
                              │
                    Pneumonia / Normal
```

### Components

| Component | Model | Frozen? | Output |
|:---|:---|:---:|:---:|
| Image Encoder | DenseNet-121 + CBAM | ✅ Yes (Phase 1 weights) | 1024-d |
| Text Encoder | Bio_ClinicalBERT | ⚡ Partial (last 2 layers) | 768-d tokens |
| Cross-Attention | MultiheadAttention (8 heads) | ❌ No | 512-d |
| Metadata MLP | Linear 17→128→128→64 | ❌ No | 64-d |
| Fusion Head | MLP 1600→512→128→2 | ❌ No | 2 logits |

---

## Project Structure

```
PneumoFusionNet/
├── mimic/
│   ├── main/
│   │   ├── Phase-1/
│   │   │   └── Phase-1.1v4-crossval_tta_PA.ipynb        ← Phase 1 (half dataset)
│   │   ├── Phase-2/
│   │   │   └── Phase-2v2-multimodal_improved_PA.ipynb   ← Phase 2 (half dataset)
│   │   ├── Phase-3/
│   │   │   └── Phase-3-triple_fusion_PA.ipynb           ← Phase 3 (half dataset)
│   │   ├── Scaleup/
│   │   │   ├── Phase-1.1v4-crossval_tta_PA_Scaleup.ipynb    ← Phase 1 (scaleup)
│   │   │   ├── Phase-2v2-multimodal_improved_PA_Upscaled.ipynb ← Phase 2 (scaleup)
│   │   │   └── Phase-3-triple_fusion_PA_Scaleup.ipynb   ← Phase 3 (scaleup) ⭐
│   │   ├── dataset/
│   │   │   ├── mimic_paired_dataset_phase2.csv          ← Image + Text manifest
│   │   │   ├── phase3_clinical_data.csv                 ← Clinical features (half)
│   │   │   ├── phase3_paired_scaleup_final.csv          ← Full merged dataset ⭐
│   │   │   ├── build_phase3_scaleup_clinical.py         ← MIMIC-IV extractor
│   │   │   ├── build_phase3_scaleup_final.py            ← Dataset merger
│   │   │   └── validate_phase3_scaleup.py               ← 31-check validator
│   │   └── outputs/
│   │       ├── Phase_1.1v4_PA_crossval_scaleup/
│   │       │   ├── best_model_fold5.pth                 ← Best Phase 1 checkpoint
│   │       │   └── lung_bboxes.csv                      ← Lung bounding boxes
│   │       ├── Phase_2v2_Scaleup/
│   │       │   └── best_v2_model.pth                    ← Phase 2 checkpoint
│   │       ├── Phase_3_triple_fusion_scaleup/
│   │       │   ├── best_p3_model.pth                    ← Phase 3 checkpoint ⭐
│   │       │   ├── phase3_results.json                  ← All metrics
│   │       │   ├── p3_roc_curve.png
│   │       │   ├── p3_confusion_matrices.png
│   │       │   ├── p3_training_curves.png
│   │       │   ├── p3_feature_importance.png
│   │       │   └── p3_full_pipeline_comparison.png
│   │       └── full_pipeline_6way_comparison.png        ← All 6 notebooks chart
```

---

## Dataset

| Property | Half Dataset | Scaleup Dataset |
|:---|:---:|:---:|
| **Source** | MIMIC-CXR | MIMIC-CXR |
| **View** | PA (Posteroanterior) only | PA only |
| **Total images** | ~1,857 | ~3,763 |
| **Normal** | 928 | 1,872 |
| **Pneumonia** | 929 | 1,891 |
| **Balance** | 1:1 | 1:1 |
| **Clinical features** | 16 | 17 (+ has_vitals flag) |

### Clinical Features (17 total)
```
Demographics : age, gender_M, is_deceased
Vitals       : heart_rate, respiratory_rate, spo2,
               systolic_bp, diastolic_bp, temperature_f
Labs         : wbc, hemoglobin, hematocrit, creatinine,
               crp, alk_phos, albumin
Flag         : has_vitals  (1 = vitals present, 0 = imputed)
```

> **Data source**: MIMIC-CXR v2.0 (images + reports) + MIMIC-IV (clinical data)
> Access requires [PhysioNet credentialing](https://physionet.org/settings/credentialing/)

---

## Pipeline Phases

### Phase 1 — Image Classification
- **Model**: DenseNet-121 + CBAM channel/spatial attention
- **Training**: 5-fold cross-validation, TTA at inference
- **Best fold**: Fold 5 (AUC = 0.8500)
- **Notebooks**: `Phase-1.1v4-crossval_tta_PA.ipynb` / `..._Scaleup.ipynb`

### Phase 2 — Image + Radiology Text
- **Image**: Frozen Phase 1 Fold 5 encoder
- **Text**: Bio_ClinicalBERT — FINDINGS + HISTORY (leakage-safe, no IMPRESSION)
- **Fusion**: CrossAttention (image queries text) → 512-d
- **Training**: Focal Loss (γ=2) + Mixup (α=0.2)
- **Notebooks**: `Phase-2v2-multimodal_improved_PA.ipynb` / `..._Upscaled.ipynb`

### Phase 3 — Triple Fusion ⭐
- **Adds**: Clinical metadata MLP on top of Phase 2
- **Metadata**: 17 features, median-imputed per split (no leakage)
- **Fusion**: concat[img(1024) + CrossAttn(512) + meta(64)] = 1600-d
- **Warm start**: Phase 2 cross-attention weights transferred
- **Notebooks**: `Phase-3-triple_fusion_PA.ipynb` / `..._Scaleup.ipynb`

---

## Results

### Progressive Improvement

```
                    AUC
Phase 1 (image)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░  0.826
Phase 2 (+text)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░  0.946   +12%
Phase 3 (+meta)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░  0.989   +4.4%
```

### Phase 3 Threshold Analysis (Scaleup)

| Threshold | Strategy | Accuracy | Sensitivity | Specificity |
|:---|:---|:---:|:---:|:---:|
| 0.500 | Default | 91.2% | 88.6% | 93.8% |
| Youden-J | Best balance | 91.7% | 89.1% | 94.3% |
| Clinical | ≥90% sensitivity | — | ≥90.0% | adjusted |

### Top Clinical Features (Permutation Importance)
Ranked by AUC drop when feature is shuffled:
1. `creatinine` — kidney function marker
2. `age` — strong demographic predictor
3. `wbc` — white blood cell count (infection marker)
4. `hemoglobin` — oxygenation capacity
5. `has_vitals` — whether ICU vitals available

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PneumoFusionNet.git
cd PneumoFusionNet

# Create virtual environment
python -m venv venv_PneumoFusionNet
venv_PneumoFusionNet\Scripts\activate   # Windows
# source venv_PneumoFusionNet/bin/activate  # Linux/Mac

# Install dependencies
pip install torch torchvision
pip install transformers pandas numpy scikit-learn
pip install matplotlib tqdm pillow opencv-python
pip install jupyter notebook
```

### Dependencies

| Package | Purpose |
|:---|:---|
| `torch` + `torchvision` | DenseNet, training |
| `transformers` | Bio_ClinicalBERT |
| `pandas` + `numpy` | Data processing |
| `scikit-learn` | Metrics, scaling, splitting |
| `matplotlib` | Visualizations |
| `Pillow` + `cv2` | Image loading |
| `tqdm` | Progress bars |

---

## How to Run

### Full Pipeline (Fresh)
```
1. Run Phase 1 notebook  → trains DenseNet + CBAM, saves fold5 checkpoint
2. Run Phase 2 notebook  → loads fold5, adds ClinicalBERT fusion
3. Run Phase 3 notebook  → loads Phase 2 weights, adds clinical MLP
```

### Quick Resume (After Kernel Restart)
```python
# Run in Phase 3 notebook after Config cell:
# The Quick Resume cell will load:
#   ✅ train_df, val_df, test_df  (preprocessed splits)
#   ✅ scaler + train_medians     (fitted on train only)
#   ✅ best_p3_model.pth          (trained weights)
# Then go directly to Cell 12 (Test Evaluation)
```

### Dataset Preparation (Phase 3 Scaleup only)
```bash
# Build clinical features from MIMIC-IV raw tables
python main/dataset/build_phase3_scaleup_clinical.py

# Merge sources and create final CSV
python main/dataset/build_phase3_scaleup_final.py

# Validate (31 checks)
python main/dataset/validate_phase3_scaleup.py
```

---

## Clinical Notes

- **No data leakage**: Patient-level train/val/test splits — same patient never appears in multiple splits
- **Leakage-safe text**: IMPRESSION section removed from reports (contains diagnosis labels). Only FINDINGS + HISTORY used
- **Imputation**: Missing clinical values filled with **training-set median only** — val/test use train medians
- **Threshold tuning**: Youden-J threshold for balanced performance; clinical threshold configurable for ≥90% sensitivity deployment
- **has_vitals flag**: Allows model to distinguish truly-measured vs imputed vital signs

---

## Citation

If you use this work, please cite:

```bibtex
@misc{pneumofusionnet2026,
  title     = {PneumoFusionNet: Triple-Fusion Multimodal Pneumonia Detection
               from Chest X-Rays, Radiology Reports, and Clinical Metadata},
  author    = {Yadav, Ashutosh},
  year      = {2026},
  note      = {GitHub repository},
  url       = {https://github.com/yourusername/PneumoFusionNet}
}
```

---

## License

This project is licensed under the MIT License.
MIMIC-CXR data access requires [PhysioNet credentialing](https://physionet.org/settings/credentialing/) — data cannot be redistributed.

---

<div align="center">

**Built with PyTorch · Bio_ClinicalBERT · MIMIC-CXR**

*PneumoFusionNet — Ashutosh Yadav · 2026*

</div>
