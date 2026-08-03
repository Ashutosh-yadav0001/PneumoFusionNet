# 🫁 PneumoFusionNet: Multimodal Deep Learning for Pneumonia Detection on MIMIC-CXR

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 1.12.1](https://img.shields.io/badge/PyTorch-1.12.1-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/docs/transformers/index)
[![Dataset](https://img.shields.io/badge/PhysioNet-MIMIC--CXR--JPG-lightgrey)](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Changelog](https://img.shields.io/badge/Changelog-v1.0.0-informational)](CHANGELOG.md)
[![Contributing](https://img.shields.io/badge/Contributing-welcome-brightgreen)](CONTRIBUTING.md)

> **PneumoFusionNet** is an explainable, multimodal deep learning framework for binary pneumonia diagnosis. Aligned with state-of-the-art medical AI literature (*Frontiers in Physiology, 2025*), the pipeline fuses high-resolution visual embeddings from chest X-rays with text representations from raw radiology reports (Bio_ClinicalBERT) and clinical metadata (demographics, vitals, lab values). 
> 
> 🏆 **Phase 2v2 SOTA**: **0.9490 AUC** | **91.37% Sensitivity** | **88.63% Accuracy**
> 
> 🚀 **Phase 3 BEST**: **0.9890 AUC** | **89.1% Sensitivity** | **94.3% Specificity** | **91.7% Accuracy** (Triple Fusion, ~3,763 images)

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
Phase 1: Vision Backbone ──► Phase 2: Multimodal (Image + Text) ──► Phase 3: Triple Fusion (Image + Text + Lab/Vitals)
```

1. **Phase 1 (Visual Classifier)**: Pre-trained visual backbones (ResNet50 / DenseNet-121) augmented with Global Context Spatial Attention (GCSA / CBAM), Depthwise Separable Convolutions (DSC), Contrast Limited Adaptive Histogram Equalization (CLAHE), and Test-Time Augmentation (TTA).
2. **Phase 2 (Multimodal Cross-Attention Fusion)**: Fuses visual feature maps with domain-specific text embeddings from Bio_ClinicalBERT via 8-head Multihead Cross-Attention, optimized with Focal Loss and embedding Mixup.
3. **Phase 3 (Triple Fusion)**: Integrates 16 clinical metadata variables (demographics, vital signs, and laboratory values from MIMIC-IV) alongside visual and textual modalities.

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

    subgraph Modality 3: Clinical Metadata Branch (Phase 3)
        ClinicalData[16 Clinical Features\nVitals + Labs + Demographics] --> MetaMLP[Metadata MLP Encoder]
        MetaMLP --> MetaEmb[64-d Metadata Embedding]
    end

    subgraph Decision Head
        FusedEmb & MetaEmb --> ConcatLayer[Feature Concatenation]
        ConcatLayer --> FocalHead[Classification MLP Head\nFocal Loss gamma=2.0]
        FocalHead --> Output[Binary Pneumonia Classifier]
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

### Master Experimental Benchmark

| Phase / Model | Modality | Key Technical Enhancements | Test AUC | Accuracy | Sensitivity | Specificity | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 1 Baseline** | Image Only | Custom ResNet50 + GCSA (139 pilot) | 0.6667 | 76.19% | — | — | Early baseline |
| **Phase 1.1 Balanced** | Image Only | ResNet50 + GCSA (Balanced cohort) | 0.9167 | 85.71% | — | — | Small balanced subset |
| **Phase 1.1 Scale-Up** | Image Only | Pilot Arch (1,989 PA images) | 0.7126 | 66.21% | — | — | Raw unoptimized scaleup |
| **Phase 1.1v4 CrossVal** | Image Only | DenseNet-121 + CBAM + CLAHE + 5-Fold + TTA | 0.8445 | 77.60% | 79.50% | 76.00% | Zero patient leakage CV |
| **Phase 1.1v5 Advanced** | Image Only | DenseNet-121 + Batch Size 16 + Youden-J | 0.8591 | 77.90% | 76.40% | 79.30% | Peak image-only ceiling |
| **Phase 2 v1 Concat** | Image + Text | Frozen ClinicalBERT + FINDINGS text | 0.9109 | 85.30% | 80.60% | 89.40% | +6.6% AUC jump over image |
| **Phase 2 v2 SOTA 🏆** | **Image + Text** | **Cross-Attention + Unfrozen BERT + Focal Loss + FINDINGS+HISTORY** | **0.9490** | **88.63%** | **91.37%** | **86.30%** | **Publication-grade SOTA** |
| **Phase 3 Half (~1,857)** | Image + Text + Metadata | 16 Clinical Features + Warm-start from Phase 2 | **0.9841** | **94.7%** | **94.9%** | **94.5%** | Triple Fusion |
| **Phase 3c (WBC-Only)** | Image + Text + Metadata | 1 Clinical Feature (WBC scalar MLP) | **0.9712** | **93.1%** | **92.3%** | **94.0%** | Single lab feature ablation |
| **Phase 3 Scaleup 🚀** | **Image + Text + Metadata** | **17 Clinical Features (~3,763 images)** | **0.9890** | **91.7%** | **89.1%** | **94.3%** | **Best overall AUC** |

### 🚀 Phase 2v2 Key Breakdown

| Technique | Architectural Action | Impact / Achievement |
| :--- | :--- | :--- |
| **Expanded Context** | Switched input text from `FINDINGS` $\rightarrow$ `FINDINGS` + `HISTORY` | **+3.8% AUC** (jumped to 0.9490) |
| **Task Fine-Tuning** | Unfrozen last 2 layers of Bio_ClinicalBERT ($lr=10^{-5}$) | **Cross-modal feature alignment** |
| **Cross-Attention** | 8-Head Multihead Attention (Image queries Text) | **+3.3% Test Accuracy** (reached 88.63%) |
| **Focal Loss ($\gamma=2.0$)** | Replaced standard Cross-Entropy loss | **+10.8% Sensitivity** (leapt to 91.37%) |
| **Grad-CAM Visuals** | Feature mapping on GCSA/CBAM attention layers | **Interpretable pulmonary heatmaps** |

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
│   ├── main.tex                               # IEEE Conference Paper (LaTeX source)
│   ├── term_project_report_FINAL.html         # Formatted HTML project report
│   └── figures/                               # Architecture diagrams and evaluation figures
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

## 📄 Documentation & Media Assets

All project deliverables and presentations are included in the [`docs/`](docs/) directory:

- 📄 **IEEE Conference Paper**: [`docs/main.tex`](docs/main.tex) - Full LaTeX source formatted in standard IEEE style (`spconf.sty`).
- 📝 **Term Project Report**: [`docs/term_project_report.md`](docs/term_project_report.md) & [`docs/term_project_report_FINAL.html`](docs/term_project_report_FINAL.html).
- 📊 **Presentation Slides**: [`docs/PneumoFusionNet_Presentation.pptx`](docs/PneumoFusionNet_Presentation.pptx).
- 🎥 **Video Presentation**: [`docs/PneumoFusionNet_Presentation.mp4`](docs/PneumoFusionNet_Presentation.mp4) (404 MB complete audio/video demonstration).
- 🎙️ **Video Script**: [`docs/video_script_FINAL.md`](docs/video_script_FINAL.md).
- 📈 **Experiment Summary Report**: [`docs/Summary till now (05 july).md`](docs/Summary%20till%20now%20%2805%20july%29.md) - Deep dive into every iteration from Phase 1.1 to Phase 2v2.

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

If you use PneumoFusionNet in your research, please cite:

```bibtex
@article{yadav2025pneumofusionnet,
  title={PneumoFusionNet: Explainable Multimodal Deep Learning for Pneumonia Detection from MIMIC-CXR X-Rays and Clinical Reports},
  author={Yadav, Ashutosh},
  journal={Department of Data Science and Artificial Intelligence, IIT Guwahati},
  year={2025}
}
```

---
*Built with ❤️ at IIT Guwahati.*
