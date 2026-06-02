# PneumoFusionNet — MIMIC-CXR

> **Binary Pneumonia Detection from Chest X-rays using ResNet50 + GCSA + DSC**  
> Adapting the PneumoFusionNet architecture to the MIMIC-CXR JPG dataset as part of a multimodal clinical AI research pipeline.

---

## 📋 Project Overview

This project implements a deep learning pipeline for **automated pneumonia detection** from chest X-ray images using the **MIMIC-CXR JPG** dataset. It serves as **Phase 1** of a larger multimodal research framework that will integrate radiology text reports and structured clinical data.

The model is adapted from **PneumoFusionNet V2** — originally developed for the IU Chest X-ray dataset — and extended to MIMIC-CXR with:

- Binary classification: **Pneumonia vs Normal**
- Labels sourced from the **CheXpert labeller** (Pneumonia = 1.0 / No Finding = 1.0)
- Image-only classification (text + labs to be added in later phases)

---

## 🏗️ Model Architecture

```
Input X-ray (1-channel grayscale, 224×224)
        ↓
  ResNet50 Backbone (pretrained, frozen early layers)
        ↓
  Depthwise Separable Convolution (DSC) — 2048 → 1024
        ↓
  Global Channel-Spatial Attention (GCSA)
        ↓
  AdaptiveAvgPool → Flatten → [1024]
        ↓
  Dropout(0.5) → Linear(512) → ReLU → Dropout(0.3) → Linear(2)
        ↓
  Pneumonia / Normal
```

| Component | Role |
|---|---|
| **ResNet50** | Deep feature extraction (ImageNet pretrained) |
| **DSC** | Efficient feature compression (2048 → 1024) |
| **GCSA** | Attention on clinically relevant regions |

---

## 📂 Project Structure

```
mimic/
├── mimic_pilot/
│   ├── build_dataset.py              # Builds mimic_dataset.csv from P10 folder
│   ├── mimic_dataset.csv             # 139 labeled samples (auto-generated)
│   ├── mimic_image_classifier.ipynb  # Phase 1 training notebook
│   └── outputs/                      # Saved models, plots (git-ignored)
│       ├── best_pneumofusion_mimic.pth
│       ├── image_features.pt          # 1024-d embeddings for Phase 2
│       ├── train_loss.png
│       ├── confusion_matrix.png
│       └── roc_curve.png
├── MIMIC_CXR_JPG_P1/                 # Raw images (git-ignored)
├── mimiciv/                          # Clinical data (git-ignored)
├── mimic-cxr-2.0.0-chexpert.csv     # CheXpert labels (git-ignored)
├── fphys-16-1512835.pdf              # Reference paper (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | MIMIC-CXR JPG (PhysioNet) |
| **Folder** | P10 (partial download) |
| **Total images** | 595 JPGs |
| **Usable samples** | **139** (clean binary labels only) |
| **Pneumonia** | 21 samples |
| **Normal** | 118 samples |
| **Split** | 80% Train / 20% Test (stratified) |

### Label Rules (from CheXpert CSV)

| CheXpert Value | Label |
|---|---|
| `Pneumonia = 1.0` | 1 — Pneumonia |
| `No Finding = 1.0` | 0 — Normal |
| Uncertain (`-1.0`) or other conditions | **Excluded** |

---

## ⚙️ Setup

### Prerequisites
- Python 3.8
- CUDA-enabled GPU (tested on RTX 3050, CUDA 11.2)

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 2. Install PyTorch (CUDA 11.3 — compatible with CUDA 11.2)
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

### 3. Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Dataset CSV
```bash
python mimic_pilot/build_dataset.py
```

---

## 🚀 Training

Open the notebook and run all cells:

```bash
# Register the venv as a Jupyter kernel
python -m ipykernel install --user --name=mimic_env --display-name "MIMIC (Python 3.8 CUDA)"

# Launch Jupyter
jupyter lab
```

Then open `mimic_pilot/mimic_image_classifier.ipynb` and select kernel **"MIMIC (Python 3.8 CUDA)"**.

### Hyperparameters

| Parameter | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 8 |
| Epochs | 30 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (weighted) |

---

## 📈 Outputs

After training, the following are saved to `mimic_pilot/outputs/`:

| File | Description |
|---|---|
| `best_pneumofusion_mimic.pth` | Best model weights |
| `image_features.pt` | 1024-d image embeddings (all 139 samples) |
| `train_loss.png` | Training loss curve |
| `confusion_matrix.png` | Test set confusion matrix |
| `roc_curve.png` | ROC curve with AUC |

---

## 🗺️ Roadmap

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ In Progress | Image-only binary classifier |
| **Phase 2** | 🔜 Planned | Add radiology text (BioBERT encoder) |
| **Phase 3** | 🔜 Planned | Add structured clinical data (MIMIC-IV labs) |
| **Phase 4** | 🔜 Planned | Multi-label classification (CheXpert 14 conditions) |

---

## 📄 Reference Paper

> *"Multimodal Deep Learning for Pneumonia Detection from Chest X-rays"*  
> [`fphys-16-1512835.pdf`](fphys-16-1512835.pdf)

---

## 👤 Author

**IIT Guwahati**  
B.Sc. in Data Science & Artificial Intelligence  
📧 ay346185@gmail.com

---

## ⚠️ Data Access

MIMIC-CXR is a restricted dataset. Access requires:
1. Completing CITI training
2. Signing the PhysioNet Data Use Agreement
3. [PhysioNet Access →](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
