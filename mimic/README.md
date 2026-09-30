<div align="center">

# 🫁 PneumoFusionNet

### Multimodal Pneumonia Detection via Chest X-Ray, Radiology Text, and White Blood Cell Fusion

**Ashutosh Yadav**  
*Mehta Family School of Data Science and Artificial Intelligence*  
*Indian Institute of Technology Guwahati, Assam, India*  
`ashutosh.yadav@op.iitg.ac.in`

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-MIMIC--CXR--JPG-green?style=flat-square)](https://physionet.org/content/mimic-cxr-jpg/)
[![MIMIC-IV](https://img.shields.io/badge/Labs-MIMIC--IV-blueviolet?style=flat-square)](https://physionet.org/content/mimiciv/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 🏆 Key Empirical Milestones

| Phase / Model | Modality | AUC | Accuracy | Sensitivity | Specificity | Threshold / Protocol |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Phase 1 (CV)** | CXR Only (DenseNet-121 + CBAM) | **0.826** | 76.3% | 71.0% | 81.6% | 5-fold GroupKFold mean (AUC 0.826 ± 0.017) |
| **Phase 2v2** | CXR + Text (Bio_ClinicalBERT) | **0.946** | 87.8% | 86.6% | 89.1% | Leakage-free text, Youden-J ($\tau=0.568$) |
| **Phase 3c (Main)** | **CXR + Text + WBC** | **0.971** | **92.9%** | **92.3%** | **93.6%** | **Youden-J Optimal ($\tau=0.559$)** |
| *Phase 3c Triage* | CXR + Text + WBC | 0.971 | 92.4% | **94.7%** | 90.0% | Emergency Triage Calibration ($\tau=0.500$) |
| *Phase 3c Target* | CXR + Text + WBC | 0.971 | 92.7% | 91.2% | **94.3%** | High-Specificity Target ($\tau=0.575$) |

> 🎯 **Point-of-Care Emergency Workflow**: Fuses chest radiograph, leakage-controlled report text, and White Blood Cell (WBC) count—all three modalities are routinely available within **30 minutes** of emergency admission.

---

## 📋 Table of Contents

- [Abstract & Clinical Problem](#abstract--clinical-problem)
- [Architecture & Workflow](#architecture--workflow)
- [Anti-Leakage Protocol](#anti-leakage-protocol)
- [Dataset Composition](#dataset-composition)
- [Pipeline Progression & Ablation](#pipeline-progression--ablation)
- [Operating Threshold Analysis](#operating-threshold-analysis)
- [WBC Scalar Fusion vs. Full Panel](#wbc-scalar-fusion-vs-full-panel)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Citation](#citation)
- [License & Data Access](#license--data-access)

---

## Abstract & Clinical Problem

### Clinical Problem
Chest radiography (CXR) interpretation is constrained by severe visual overlap between pneumonia manifestations (consolidation, ground-glass opacities, interstitial patterns) and non-infectious pulmonary opacities (atelectasis, pleural effusion, edema). Furthermore, imaging alone cannot quantify host systemic inflammatory response.

### Multimodal Approach
**PneumoFusionNet** evaluates **3,763 posteroanterior (PA) radiographs** from MIMIC-CXR-JPG coupled with clinical notes and laboratory records from MIMIC-IV. The framework integrates:
1. **Domain-Pretrained DenseNet-121 Backbone** with Convolutional Block Attention Module (CBAM) producing a 1024-d image embedding.
2. **Bio_ClinicalBERT Encoder** processing diagnostic-free `FINDINGS` and `HISTORY` text via 8-head cross-attention (producing a 512-d text context).
3. **Dedicated Non-Linear MLP** ($1 \to 128 \to 128 \to 64$) projecting point-of-care White Blood Cell (WBC) count into a 64-d clinical embedding.

---

## Architecture & Workflow

### Phase 3c — Triple Fusion Architecture

```
                 Image Branch                                 Text Branch                              WBC Branch
         ┌──────────────────────────┐                 ┌──────────────────────────┐             ┌──────────────────────────┐
         │     CXR Image (PA view)  │                 │    Radiology Report:     │             │     WBC Count (scalar)   │
         │         224 × 224        │                 │   FINDINGS + HISTORY     │             │    e.g., 7.6 × 10³/µL    │
         └─────────────┬────────────┘                 └─────────────┬────────────┘             └─────────────┬────────────┘
                       │                                            │                                        │
             DenseNet-121 (xrv)                              Anti-Leakage Parser                            MLP
            (Domain Pretrained)                                     │                            1 → 128 → 128 → 64
                       │                                     Bio_ClinicalBERT                        (ReLU, Dropout)
                 CBAM Attention                             (top 2 unfrozen layers)                          │
                       │                                            │                                        │
                 Image Embedding                             Text Token Sequence                        WBC Embedding
                    (1024-d)                                      (768-d)                                 (64-d)
                       │                                            │                                        │
             z_img ∈ ℝ¹⁰²⁴                                 z_text ∈ ℝ⁷⁶⁸×N                           z_wbc ∈ ℝ⁶⁴
                       │                                            │                                        │
                       └────────────────────┬───────────────────────┘                                        │
                                            │                                                                │
                                   8-Head Cross-Attention                                                    │
                                   (Q: image, K/V: text)                                                     │
                                      Output: 512-d                                                          │
                                            │                                                                │
                                            └───────────────────────┬────────────────────────────────────────┘
                                                                    │
                                                              Concatenation
                                                 [1024 + 512 + 64] = 1600-d Fused Vector
                                                                    │
                                                              MLP Classifier
                                                         1600 → 512 → 128 → 2
                                                         (LayerNorm, Focal Loss)
                                                                    │
                                                          Normal / Pneumonia
```

### Module Specifications

| Branch / Module | Architecture & Pretraining | Trainable Params / Status | Output Vector |
|:---|:---|:---:|:---:|
| **Image Encoder** | DenseNet-121 + CBAM (pretrained on 500k CXRs via TorchXRayVision) | ❄️ Frozen downstream | $\mathbf{v}_{\text{img}} \in \mathbb{R}^{1024}$ |
| **Text Encoder** | Bio_ClinicalBERT (pretrained on MIMIC-III clinical notes) | ⚡ Top 2 layers unfrozen ($\eta_{\text{BERT}}=10^{-5}$) | $\mathbf{T} \in \mathbb{R}^{B \times 256 \times 768}$ |
| **Cross-Attention** | 8-head MultiheadAttention ($d_k = 64$, image queries text) | ⚡ Trainable ($\eta_{\text{fusion}}=2 \times 10^{-4}$) | $\mathbf{c}_{\text{cross}} \in \mathbb{R}^{512}$ |
| **WBC Encoder** | 3-Layer MLP ($1 \to 128 \to 128 \to 64$) with ReLU & Dropout | ⚡ Trainable ($\eta_{\text{wbc}}=10^{-3}$) | $\mathbf{e}_{\text{wbc}} \in \mathbb{R}^{64}$ |
| **Classifier Head** | LayerNorm(1600) $\to$ Dropout(0.4) $\to$ 512 $\to$ 128 $\to$ 2 | ⚡ Trainable ($\text{Focal Loss}, \gamma=2.0$) | $2 \text{ logits}$ |

---

## Anti-Leakage Protocol

A critical contribution of PneumoFusionNet is the enforcement of a strict **anti-leakage protocol** to prevent diagnostic label contamination from radiology reports:

1. **Impression & Conclusion Removal**: The `IMPRESSION` and `CONCLUSION` sections—which explicitly state the radiologist's final diagnosis—are stripped before tokenisation. Only `FINDINGS` and `HISTORY` are retained.
2. **Keyword Redaction**: Diagnostic trigger words and phrases in the remaining text (e.g., `"pneumonia"`, `"pneumonic"`, `"compatible with"`, `"consistent with"`, `"normal study"`, `"no acute"`, `"no finding"`, `"no significant"`) are replaced with `[REDACTED]` using regular expressions.
3. **Patient-Level Splitting**: Partitioning strictly groups by patient identifier (`subject_id`) using `GroupKFold` / `GroupShuffleSplit` with automated verification ensuring zero patient overlap:
   $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
4. **PA-Only View Filtering**: Posteroanterior (PA) radiographs were exclusively retained. Pilot experiments demonstrated that anteroposterior (AP) views strongly correlated with patient acuity (bedridden vs. ambulatory shortcut), creating an acquisition-bias shortcut.
5. **Train-Only Imputation**: Approximately 15% missing WBC values are imputed using the **training-set median exclusively**. The exact same value is applied to validation and test partitions to guarantee zero data leakage.

---

## Dataset Composition

Experiments were conducted on **MIMIC-CXR-JPG v2.0.0** linked with **MIMIC-IV** laboratory records (`labevents` item IDs 51300/51301) via patient identifiers and study date proximity ($\pm 2$ days).

```
MIMIC-CXR Scaleup Cohort (N = 3,763 PA Radiographs)
├── Normal:    1,872 (49.7%)
└── Pneumonia: 1,891 (50.3%)

Data Partitioning (Patient-Level Grouping):
├── Train: 2,633 samples (~70%)
├── Val:     565 samples (~15%)
└── Test:    565 samples (~15%)
```

---

## Pipeline Progression & Ablation

PneumoFusionNet was evaluated progressively across modalities on the 3,763-sample Scale-Up Cohort:

```
                                      AUC
Phase 1 (CXR Only)       ████████████████████░░░░░  0.826
Phase 2v2 (CXR + Text)   ███████████████████████░░  0.946  (+12.0 pp)
Phase 3c (CXR+Text+WBC)  █████████████████████████  0.971  (+2.5 pp over Phase 2)
```

### Table I: Modality Progression Metrics ($N_{\text{Scaleup}} = 3,763$)

| Phase / Model | Modality | AUC | Accuracy | Sensitivity | Specificity | Key Architectural Advantage |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **Phase 1 (CV)** | CXR Only | 0.826 | 76.3% | 71.0% | 81.6% | DenseNet-121 + CBAM, 5-fold GroupKFold, TTA |
| **Phase 2v2** | CXR + Text | 0.946 | 87.8% | 86.6% | 89.1% | Bio_ClinicalBERT + 8-Head Cross-Attention |
| **Phase 3c** | **CXR + Text + WBC** | **0.971** | **92.9%** | **92.3%** | **93.6%** | **Triple Fusion + Non-linear WBC Embedding** |

*Note: Phase 1 ImageNet-pretrained ResNet-50 pilot failed to transfer to grayscale radiological textures (AUC dropped to 0.713 on scaled data).*

---

## Operating Threshold Analysis

Decision thresholds were evaluated on the held-out test set ($N_{\text{test}}=565$) under three clinical deployment strategies:

### Table II: Phase 3c Threshold Deployment Strategies

| Strategy | Threshold ($\tau$) | Accuracy | Sensitivity | Specificity | Clinical Utility |
|:---|:---:|:---:|:---:|:---:|:---|
| **Default** | $\tau = 0.500$ | 92.4% | **94.7%** | 90.0% | **Emergency Triage**: Maximises sensitivity to avoid missed diagnoses. |
| **Youden-J (Optimal)** | $\tau = 0.559$ | **92.9%** | **92.3%** | **93.6%** | **Balanced Decision**: Maximises Youden's $J$ statistic ($J = \text{TPR} - \text{FPR}$). |
| **Clinical Target** | $\tau = 0.575$ | 92.7% | 91.2% | **94.3%** | **High Specificity**: Minimises false positive alerts while keeping sensitivity $>90\%$. |

---

## WBC Scalar Fusion vs. Full Panel

A key design finding of PneumoFusionNet is the power of a **single point-of-care laboratory scalar (WBC)** compared to a full 17-feature clinical panel:

| Model Configuration | Input Modalities | Features Used | AUC | Sensitivity | Turnaround Time |
|:---|:---|:---:|:---:|:---:|:---:|
| **Full Clinical Panel** | CXR + Text + 17 Clinical Vars | Demographics, Vitals, Labs | 0.969 | 89.1% | 1 – 4 hours |
| **Phase 3c (PneumoFusionNet)** | **CXR + Text + WBC Scalar** | **WBC Count Only** | **0.971** | **92.3%** | **< 30 minutes** |

### Why WBC Fusion is Clinically Superior:
1. **Dramatically Faster Emergency Workflow**: Complete Blood Count (CBC) providing WBC count is available within **15 minutes** of patient presentation. The CXR + Text + WBC triad can be evaluated within **30 minutes of emergency admission**, whereas full lab panels require hours.
2. **Higher Sensitivity & Discrimination**: WBC scalar non-linear projection ($1 \to 128 \to 128 \to 64$) captures critical systemic inflammation thresholds (borderline leukocytosis vs. severe acute infection) without suffering from the missingness noise ($\sim 78\%$ missing vitals in non-ICU patients) present in full electronic health record panels.

---

## Project Structure

```
PneumoFusionNet/
├── mimic/
│   ├── main/
│   │   ├── Phase-1/
│   │   │   └── Phase-1.1v4-crossval_tta_PA.ipynb            ← Phase 1 (DenseNet+CBAM CV)
│   │   ├── Phase-2/
│   │   │   └── Phase-2v2-multimodal_improved_PA.ipynb       ← Phase 2 (CXR + Bio_ClinicalBERT)
│   │   ├── Phase-3/
│   │   │   ├── Phase-3-triple_fusion_PA.ipynb               ← Phase 3 (Half cohort)
│   │   │   └── Phase-3-triple_fusion_PA_Scaleup.ipynb       ← Phase 3c Main Model ⭐
│   │   ├── Scaleup/
│   │   │   └── Phase-3-triple_fusion_PA_11_scaleup_features.ipynb ← 11-feature scaleup
│   │   ├── dataset/
│   │   │   ├── phase3_paired_scaleup_final.csv              ← Scaleup paired manifest ⭐
│   │   │   ├── phase2_reports_no_impression.csv             ← Leakage-filtered text
│   │   │   ├── phase3_clinical_data.csv                     ← Clinical features
│   │   │   └── build_phase3_scaleup_final.py                ← Dataset generation script
│   │   └── outputs/
│   │       ├── Phase_1.1v4_PA_crossval_scaleup/
│   │       │   ├── best_model_fold5.pth                     ← Phase 1 checkpoint
│   │       │   └── lung_bboxes.csv                          ← Bounding box annotations
│   │       ├── Phase_2v2_Scaleup/
│   │       │   └── best_v2_model.pth                        ← Phase 2 checkpoint
│   │       └── Phase_3_triple_fusion_scaleup_11features/
│   │           ├── best_p3_model.pth                        ← Phase 3c checkpoint ⭐
│   │           ├── phase3_results.json                      ← Comprehensive metrics
│   │           ├── p3_roc_curve.png                         ← ROC curve figure
│   │           ├── p3_confusion_matrices.png                ← Threshold confusion matrices
│   │           ├── p3_feature_importance.png                ← Permutation feature importance
│   │           └── p3_training_curves.png                   ← Convergence plots
│   └── README.md
```

---

## Setup & Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Ashutosh-yadav0001/PneumoFusionNet.git
cd PneumoFusionNet/mimic

# Create and activate Python virtual environment
python -m venv venv_PneumoFusionNet
venv_PneumoFusionNet\Scripts\activate      # Windows
# source venv_PneumoFusionNet/bin/activate  # Linux / macOS

# Install core dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install transformers torchxrayvision opencv-python pillow pandas numpy scikit-learn matplotlib seaborn tqdm
```

---

## How to Run

### Sequential Pipeline Execution

1. **Phase 1 (CXR Encoder)**: Run `main/Phase-1/Phase-1.1v4-crossval_tta_PA.ipynb` to train the DenseNet-121 + CBAM backbone with 5-fold cross-validation and generate lung bounding box crops.
2. **Phase 2 (Text Fusion)**: Run `main/Phase-2/Phase-2v2-multimodal_improved_PA.ipynb` to train 8-head cross-attention with Bio_ClinicalBERT using leakage-filtered text.
3. **Phase 3c (Triple Fusion Main Model)**: Run `main/Scaleup/Phase-3-triple_fusion_PA_11_scaleup_features.ipynb` to train the full multimodal triad (CXR + Text + WBC) with warm-started cross-attention weights.

---

## Citation

If you use **PneumoFusionNet** or our anti-leakage multimodal evaluation framework in your research, please cite our paper:

```bibtex
@article{yadav2025pneumofusionnet,
  title     = {PneumoFusionNet: Multimodal Pneumonia Detection via Chest X-Ray, Radiology Text, and White Blood Cell Fusion},
  author    = {Yadav, Ashutosh},
  journal   = {Mehta Family School of Data Science and Artificial Intelligence, Indian Institute of Technology Guwahati},
  year      = {2025},
  url       = {https://github.com/Ashutosh-yadav0001/PneumoFusionNet}
}
```

---

## License & Data Access

- **Code License**: Licensed under the [MIT License](LICENSE).
- **Dataset Access**: MIMIC-CXR-JPG and MIMIC-IV are hosted on [PhysioNet](https://physionet.org/). Access requires PhysioNet credentialing (CITI Data or Specimens Only Research training and signed Data Use Agreements). Dataset files cannot be redistributed.

---

<div align="center">

**Mehta Family School of Data Science and Artificial Intelligence**  
*Indian Institute of Technology Guwahati, Assam, India*

</div>
