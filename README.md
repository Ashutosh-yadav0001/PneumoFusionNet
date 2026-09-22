# 🫁 PneumoFusionNet: Multimodal Deep Learning for Pneumonia Detection on MIMIC-CXR

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/index)
[![Dataset](https://img.shields.io/badge/PhysioNet-MIMIC--CXR--JPG-lightgrey)](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Changelog](https://img.shields.io/badge/Changelog-v1.0.0-informational)](CHANGELOG.md)
[![Contributing](https://img.shields.io/badge/Contributing-welcome-brightgreen)](CONTRIBUTING.md)

> **PneumoFusionNet** is an explainable, multimodal deep learning framework for binary pneumonia diagnosis. Validated on 3,763 PA-view studies from the **MIMIC-CXR** and **MIMIC-IV** databases, the pipeline progressively fuses domain-pretrained chest X-ray representations (DenseNet-121 + CBAM) with leakage-controlled clinical radiology text (Bio_ClinicalBERT via 8-head cross-attention) and a routine White Blood Cell (WBC) count.
> 
> 🏆 **Conference Main Model (Phase 3c: CXR + Text + WBC)**: **0.9711 Test AUC** | **92.25% Sensitivity** | **93.59% Specificity** | **92.92% Accuracy** (Youden-J, $N_{\text{test}}=565$). Reaches **94.72% Sensitivity** at emergency screening threshold ($\tau = 0.500$).
> 
> 📈 **Progressive Trajectory**: **Phase 1 Image Baseline**: 0.8258 AUC $\to$ **Phase 2v2 Image + Text**: 0.9460 AUC $\to$ **Phase 3c Image + Text + WBC**: **0.9711 AUC**.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture & Pipeline](#-architecture--pipeline)
- [Anti-Leakage Protocol](#-anti-leakage-protocol)
- [Experimental Results](#-experimental-results)
- [Repository Structure](#-repository-structure)
- [Documentation & Media Assets](#-documentation--media-assets)
- [Setup & Installation](#-setup--installation)
- [Quick Start](#-quick-start)
- [Data Access Notice](#-data-access-notice)
- [Author & Citation](#-author--citation)

---

## 📋 Project Overview

Medical diagnosis of pneumonia using chest radiography (CXR) alone is subject to visual ambiguity and inter-observer variability. Human clinicians synthesize visual findings with clinical history, lab metrics, and radiology notes. **PneumoFusionNet** models this clinical workflow through a progressive multi-phase architecture on the restricted **MIMIC-CXR** dataset:

```
Phase 1: Vision Backbone ──► Phase 2: Multimodal (Image + Text) ──► Phase 3c: Triple Fusion (Image + Text + 15-min WBC)
```

1. **Phase 1 (Visual Classifier)**: Pre-trained visual backbones (DenseNet-121 via TorchXRayVision) augmented with Convolutional Block Attention Module (CBAM), Focal Loss ($\gamma=2.0$), and Test-Time Augmentation (TTA). Reaches an image-only ceiling at **0.826 AUC** (5-fold CV).
2. **Phase 2 (Multimodal Cross-Attention Fusion)**: Fuses visual feature maps with domain-specific text embeddings from Bio_ClinicalBERT via 8-head Multihead Cross-Attention, strictly eliminating `IMPRESSION` conclusions and redacting diagnostic keywords. Jumps to **0.946 AUC** (+12.0 pp).
3. **Phase 3c (WBC Triple Fusion — Conference Focus)**: Incorporates a single point-of-care White Blood Cell (WBC) count from routine CBC testing projected through a non-linear MLP ($1 \to 128 \to 128 \to 64$). Reaches **0.9711 AUC**, **92.25% Sensitivity**, and **93.59% Specificity**, outperforming a full 17-feature EHR panel in sensitivity while remaining deliverable within 30 minutes of emergency admission.

---

## 🏗️ Architecture & Pipeline

```mermaid
graph TD
    subgraph Modality 1: Vision Branch
        Img[Chest X-ray Image] --> Preproc[CLAHE Preprocessing]
        Preproc --> VisionBackbone["DenseNet-121 / ResNet50"]
        VisionBackbone --> DSC[Depthwise Separable Conv]
        DSC --> Attention[CBAM / GCSA Spatial Attention]
        Attention --> ImgEmb[1024-d Image Embedding]
    end

    subgraph Modality 2: Clinical Text Branch
        Report[Radiology Report Text] --> LeakageFilter[Anti-Leakage Parser\nExclude IMPRESSION / Redact Keywords]
        LeakageFilter --> BERT[Bio_ClinicalBERT Encoder\nLast 2 Layers Unfrozen]
        BERT --> TextEmb[768-d Text Tokens]
    end

    subgraph Cross-Attention & Multimodal Fusion
        ImgEmb & TextEmb --> CrossAttn["8-Head Multihead Cross-Attention\n(Image queries Text)"]
        CrossAttn --> FusedEmb[512-d Multimodal Vector]
    end

    subgraph Modality 3: Laboratory Branch (Phase 3c / Main Model)
        WBCData["White Blood Cell Count (WBC)\nRoutine 15-min CBC Scalar"] --> MetaMLP["MLP Encoder\n(1 -> 128 -> 128 -> 64)"]
        MetaMLP --> MetaEmb[64-d WBC Embedding]
    end

    subgraph Decision Head
        ImgEmb & FusedEmb & MetaEmb --> ConcatLayer["Feature Concatenation\n[1024 + 512 + 64] = 1600-d Vector"]
        ConcatLayer --> FocalHead["Classification Head (MLP)\n1600 -> 512 -> 128 -> 2\nFocal Loss gamma=2.0"]
        FocalHead --> Output[Normal / Pneumonia]
        FocalHead --> Explain[Grad-CAM Heatmap Visualization]
    end
```

### Core Innovations

* **TorchXRayVision DenseNet-121 Backbone**: Pre-trained on multi-million chest X-rays to extract domain-specific radiological features.
* **Global Context Spatial Attention (GCSA) & CBAM**: Captures long-range spatial dependencies and focuses network attention on pulmonary infiltrates and consolidation.
* **Bio_ClinicalBERT Fine-Tuning**: Pre-trained on MIMIC-III clinical notes; top 2 layers are fine-tuned alongside the fusion network with separate learning rates ($10^{-5}$).
* **Cross-Attention Fusion**: Allows image features to visually query tokenized clinical text (`FINDINGS` + `HISTORY`), capturing subtle disease indicators.
* **Focal Loss ($\gamma = 2.0$) & Youden-J Thresholding**: Addresses dataset imbalance and targets clinical-grade sensitivity ($\ge 90\%$).

---

## 🔒 Anti-Leakage Protocol

Radiology report summaries (specifically `IMPRESSION` or `CONCLUSION` sections) routinely state the final clinical diagnosis. Standard multimodal models trained on raw reports suffer from severe **label leakage** (reading the written diagnosis instead of diagnosing from medical findings).

To ensure complete diagnostic integrity, our pipeline enforces strict on-the-fly text parsing:
* **Impression Removal**: The `IMPRESSION` and `CONCLUSION` sections are completely stripped before tokenization.
* **Keyword Redaction**: Specific diagnostic label triggers (e.g., explicit mentions of "pneumonia") in `FINDINGS` and `HISTORY` sections are redacted.
* **Anti-Cheating Verification**: The network is forced to correlate visual opacities in the X-ray with descriptive findings (patient symptoms, fever, cough, auscultation notes).

---

## 📊 Experimental Results

All experiments were systematically evaluated using reproducible seeds (`SEED=42`) and zero-patient-leakage splits (`GroupKFold` / stratified splits).

### 🏆 Primary Scale-Up Benchmark (3,763 PA Images — Conference Paper Table I)

Evaluated with strict zero-patient-leakage splitting on the standardized scale-up cohort:

| Phase / Model | Modality | Technical Architecture | Test AUC | Accuracy | Sensitivity | Specificity | Cohort / Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 1 (CV Scaleup)** | CXR only | DenseNet-121 + CBAM + CLAHE + Focal Loss | **0.8258 ± 0.017** | **76.3%** | **71.0%** | **81.6%** | 5-fold GroupKFold ($N=3,763$) |
| **Phase 2v2 (Scaleup)** | CXR + Text | Bio_ClinicalBERT (top 2 unfrozen) + 8-Head Cross-Attn | **0.9460** | **87.8%** | **86.6%** | **89.1%** | Youden-J ($\tau=0.568$, $N_{\text{test}}=582$) |
| **Phase 3c (WBC Scaleup) 🏆** | **CXR + Text + WBC** | **Cross-Attn + 15-min POC WBC MLP ($1\to 128\to 128\to 64$)** | **0.9711** | **92.9%** | **92.25%** | **93.59%** | **Conference Focus** ($\tau=0.559$, $N_{\text{test}}=565$) |
| **Phase 3 Full (Scaleup)** | CXR + Text + EHR | Full 17-feature EHR panel (Vitals + Labs + Demographics) | **0.9690** | **91.7%** | **89.1%** | **94.3%** | Delayed 1–4h lab panel ($N_{\text{test}}=565$) |

> 💡 **Key Clinical Finding**: Integrating a single point-of-care WBC count (ready in 15 minutes) achieves **0.9711 AUC** and outperforms the full 17-variable EHR panel in sensitivity (**92.25% vs. 89.08%**), enabling complete triage within 30 minutes of emergency presentation.

---

### 🎯 Phase 3c Operating Threshold Analysis (Conference Paper Table II)

Operating performance across different clinical deployment scenarios on the held-out test partition ($N_{\text{test}}=565$):

| Strategy | Decision Threshold ($\tau$) | Accuracy | Sensitivity | Specificity | Recommended Clinical Setting |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Emergency Screening** | **0.500** | 92.4% | **94.72%** | 90.04% | ED Triage (minimise missed infections) |
| **Youden-J (Optimal)** | **0.559** | **92.92%** | **92.25%** | **93.59%** | Balanced diagnostic decision support |
| **Clinical Target** | **0.575** | 92.74% | 91.20% | **94.31%** | Confirmatory testing (maximise specificity) |

---

### 🔬 Exploratory & Development Iterations (Subset Experiments)

| Phase / Notebook | Modality | Cohort Size | Test AUC | Accuracy | Sensitivity | Specificity | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 1 Baseline** | Image Only | 139 pilot | 0.6667 | 76.19% | — | — | Initial feasibility pilot |
| **Phase 1.1 Balanced** | Image Only | 1,854 balanced | 0.9167 | 85.71% | — | — | ResNet50 + GCSA |
| **Phase 1.1 Scale-Up Pilot** | Image Only | 1,989 PA | 0.7126 | 66.21% | — | — | Raw unoptimized scaleup |
| **Phase 1.1v5 Advanced** | Image Only | 1,989 PA | 0.8591 | 77.90% | 76.40% | 79.30% | Image-only ceiling on subset |
| **Phase 2 v1 Concat** | Image + Text | 1,989 PA | 0.9109 | 85.30% | 80.60% | 89.40% | Frozen BERT + FINDINGS text |
| **Phase 2 v2 Improved** | Image + Text | 1,989 PA ($N_{\text{test}}=299$) | 0.9490 | 88.63% | 91.37% | 86.25% | Subset run with unfrozen BERT |
| **Phase 3 Half-Dataset** | Image + Text + 16 Meta | 1,857 PA ($N_{\text{test}}=398$) | 0.9841 | 94.72% | 94.94% | 94.55% | 16-feature subset benchmark |

---

## 📁 Repository Structure

```text
PneumoFusionNet/
├── mimic/                                     # ⭐ Primary experiment directory
│   ├── main/                                  # Multi-Phase Pipeline Notebooks & Scripts
│   │   ├── Phase-1/                           # Phase 1: DenseNet-121 + CBAM visual classifiers
│   │   ├── Phase-2/                           # Phase 2: Bio_ClinicalBERT + CrossAttention fusion
│   │   ├── Phase-3/                           # Phase 3: Triple Fusion (Full Metadata & Phase 3c WBC-Only)
│   │   ├── Scaleup/                           # Scale-up experiments (~3,763 images)
│   │   ├── dataset/                           # CSV manifests & dataset build scripts
│   │   └── outputs/                           # Checkpoints, metrics, and ROC/PR plots
│   │
│   ├── 1000_dataset/                          # 1,000 Balanced MIMIC Cohort Guides & CSVs
│   ├── mimic_pilot_139/                       # Pilot 139-image cohort (Phase 1 baseline)
│   ├── mimiciv/                               # MIMIC-IV tabular EHR processing
│   └── README.md                              # MIMIC sub-folder guide
│
├── src/                                       # 🐍 Reusable Python package (pip install -e .)
│   ├── models/
│   │   ├── vision.py                          # ChannelAttention, CBAM, EnhancedPneumoNetV4, ImageEncoder
│   │   ├── text_encoder.py                    # TextEncoder (Bio_ClinicalBERT partial fine-tune)
│   │   └── fusion.py                          # CrossAttnFusionNet, MetadataEncoder, TripleFusionNet
│   ├── data/
│   │   ├── dataset.py                         # CXRDataset, MultimodalCXRDataset, TripleModalCXRDataset
│   │   └── preprocessing.py                   # CLAHE, bbox, anti-leakage text, TTA transforms
│   └── utils/
│       ├── metrics.py                         # evaluate, TTA eval, threshold selection, feat importance
│       └── training.py                        # FocalLoss, Mixup variants, train loop, optimizer builder
│
├── docs/                                      # Research papers, presentations & documentation
│   ├── API.md                                 # Full src/ module API reference
│   ├── main.tex                               # Project report (LaTeX source)
│   ├── term_project_report_FINAL.html         # Formatted HTML project report
│   └── figures/                               # Architecture diagrams and evaluation figures
│
├── report Writing/                            # 📄 Conference Paper & Submission Package
│   ├── conference_101719.tex                  # IEEE conference paper (local source)
│   ├── paper_for_editing.txt                  # Full paper text file for editing
│   ├── architecture.png                       # Publication architecture diagram
│   ├── Overleaf_Upload/                       # Self-contained Overleaf project folder
│   │   ├── main.tex                           # Overleaf main document
│   │   ├── IEEEtran.cls                       # IEEE conference class file
│   │   └── *.png                              # All figure assets
│   └── PneumoFusionNet_Overleaf.zip           # Ready-to-upload Overleaf ZIP package
│
├── model_experiments/                         # Early exploratory notebooks (IU X-Ray dataset)
├── experiment_results/                        # Saved visualisations and result artefacts
├── pyproject.toml                             # Python package config (pip install -e .)
├── requirements.txt                           # Global project dependencies
├── CHANGELOG.md                               # Full version history
├── CONTRIBUTING.md                            # Contribution guide
├── CODE_OF_CONDUCT.md                         # Community standards
├── LICENSE                                    # MIT License
└── README.md                                  # This file
```

---



## ⚙️ Setup & Installation

### Prerequisites
* **Python**: 3.10+
* **Hardware**: NVIDIA GPU with CUDA 11.3+ support (e.g., RTX 3050 / RTX 3090 / A100)

### 1. Clone Repository & Setup Environment

```bash
git clone https://github.com/Ashutosh-yadav0001/PneumoFusionNet.git
cd PneumoFusionNet

# Create Python Virtual Environment
python -m venv venv_PneumoFusionNet

# Activate Environment
# Windows:
venv_PneumoFusionNet\Scripts\activate
# Linux/macOS:
source venv_PneumoFusionNet/bin/activate
```

### 2. Install PyTorch & Dependencies

Install PyTorch compiled for CUDA 11.3+:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install core dependencies (Transformers, TorchXRayVision, scikit-learn, OpenCV, etc.):

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Lab

```bash
# (Optional) Install src/ as an editable package for notebook imports:
pip install -e .

# Register kernel in Jupyter
python -m ipykernel install --user --name=venv_PneumoFusionNet --display-name "PneumoFusionNet"

# Launch JupyterLab
jupyter lab
```

---

## ⚡ Quick Start

After installing the package (`pip install -e .`), you can import any component directly:

```python
import torch
from src.models import ImageEncoder, TextEncoder, TripleFusionNet
from src.data import TripleModalCXRDataset, get_val_transforms, build_report_text
from src.utils import find_optimal_threshold, compute_metrics, set_seed

# Reproducibility
set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build encoders
img_enc  = ImageEncoder("mimic/main/outputs/.../best_model_fold5.pth").to(device)
txt_enc  = TextEncoder().to(device)

# Build Phase 3 triple fusion model
model = TripleFusionNet().to(device)
model.load_phase2_weights("mimic/main/outputs/.../best_v2_model.pth")

# Run inference on a single image
with torch.no_grad():
    img_feat  = img_enc(images.to(device))           # (B, 1024)
    txt_feat  = txt_enc(input_ids, attn_mask)        # (B, 256, 768)
    logits    = model(img_feat, txt_feat, metadata)  # (B, 2)

# Threshold selection
threshold = find_optimal_threshold(test_labels, test_probs)
results   = compute_metrics(test_labels, test_probs, threshold=threshold)
```

---

## 🔑 Data Access Notice

**MIMIC-CXR** and **MIMIC-IV** are credentialed-access clinical databases hosted by PhysioNet. To access raw DICOM/JPG images, radiology text notes, or tabular patient records:

1. Complete the CITI Training Course ("Human Subjects Research - Data or Specimens Only").
2. Sign the PhysioNet Data Use Agreement (DUA).
3. Submit an access request on [PhysioNet MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
4. Follow our [1,000 Cohort Guide](mimic/1000_dataset/README.md) to generate matching image-text-metadata pairs.

---

## 👨‍💻 Author & Citation

**Ashutosh Yadav**  
* **Affiliation**: Indian Institute of Technology Guwahati (IIT Guwahati)  
* **Program**: B.Sc. (Honours) in Data Science & Artificial Intelligence  
* **Email**: [ashutosh@op.iitg.ac.in](mailto:ashutosh@op.iitg.ac.in) | [ay346185@gmail.com](mailto:ay346185@gmail.com)  
* **GitHub**: [@Ashutosh-yadav0001](https://github.com/Ashutosh-yadav0001)

### BibTeX Citation

If you use PneumoFusionNet in your research or baseline comparisons, please cite our conference paper:

```bibtex
@inproceedings{yadav2026pneumofusionnet,
  title={PneumoFusionNet: Multimodal Pneumonia Detection via Chest X-Ray, Radiology Text, and White Blood Cell Fusion},
  author={Yadav, Ashutosh},
  booktitle={IEEE Conference Submission},
  year={2026},
  organization={Mehta Family School of Data Science and Artificial Intelligence, Indian Institute of Technology Guwahati}
}
```

---
*Built with ❤️ at IIT Guwahati.*
