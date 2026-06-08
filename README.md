# PneumoFusionNet on MIMIC-CXR
### Explainable Multimodal Deep Learning Framework for Pneumonia Detection from MIMIC-CXR Chest X-rays and Clinical Reports

PneumoFusionNet is a state-of-the-art multimodal deep learning framework adapted for the MIMIC-CXR JPG dataset (P10 subset). The framework fuses chest X-ray images, clinical reports, and structured clinical laboratory values to diagnose binary pneumonia with high classification performance and clinical interpretability.

---

## 🚀 Multimodal Diagnosis Roadmap

The framework is structured into three phases, following the methodology described in the paper:
> *“A multi-modal deep learning solution for precise pneumonia diagnosis: the PneumoFusion-Net model”* (Frontiers in Physiology, 2025).

```
Phase 1: Image Only ──► Phase 2: Multimodal (Image + Text) ──► Phase 3: Multimodal + Lab Data
```

### 1. Phase 1 — Chest X-ray Image Classifier
- **Model**: Custom ResNet50 backbone + Depthwise Separable Convolution (DSC) + Global Context Spatial Attention (GCSA).
- **Input**: Grayscale chest X-ray images (224x224).
- **Output**: 1024-d image embeddings + Binary Pneumonia Classifier.
- **Notebook**: [Phase-1mimic_image_classifier.ipynb](file:///C:/2026/PneumoFusionNet/mimic/mimic_pilot_139/Notebooks/Phase-1mimic_image_classifier.ipynb)

### 2. Phase 2 — Multimodal Fusion (Image + Clinical Text)
- **Model**: Fuses frozen 1024-d Phase 1 image embeddings with 768-d text representations from `emilyalsentzer/Bio_ClinicalBERT` (fine-tuning the last 2 transformer layers).
- **Input**: Chest X-ray + Patient Clinical Text.
- **Anti-Leakage Design**: Completely avoids the `impression` section of the radiology report (which contains the final diagnosis, causing label leakage). Instead, it reads the report and parses **everything except the IMPRESSION section** (clinical history, technique, comparison, findings) on-the-fly.
- **Output**: 1792-d multimodal embeddings + Binary Pneumonia Classifier.
- **Notebook**: [Phase-2-FINAL_multimodal_classifier.ipynb](file:///C:/2026/PneumoFusionNet/mimic/mimic_pilot_139/Notebooks/Phase-2-FINAL_multimodal_classifier.ipynb)

### 3. Phase 3 — Multimodal + Lab Data Fusion (Upcoming)
- **Model**: Will fuse the 1792-d multimodal image-text embeddings with structured MIMIC-IV laboratory values (WBC count, CRP, procalcitonin, etc.).

---

## 📊 Experimental Results

Both phases were trained using the exact same stratified 70/15/15 train/validation/test split (`SEED=42`) to prevent split-wise data leakage. 

| Model | Test AUC | Test Accuracy | Test F1 (macro) |
| :--- | :---: | :---: | :---: |
| **Phase 1 (Image Only)** | 0.6667 | 0.7619 | 0.4300 |
| **Phase 2 (Image + Report)** | **0.7037** | **0.8095** | **0.4474** |
| **Multimodal Improvement** | **+0.0370** | **+0.0476** | **+0.0174** |

*The addition of clinical report context (history + findings) without label leakage yields significant improvements in both AUC and overall diagnostic accuracy.*

---

## 📁 Repository Structure

The code and outputs for the MIMIC pilot subset are organized under the `mimic/mimic_pilot_139/` subdirectory:

```text
PneumoFusionNet/
│
├── mimic/
│   └── mimic_pilot_139/
│       ├── dataset_139/
│       │   ├── mimic_dataset.csv               # Baseline image dataset
│       │   └── mimic_multimodal_dataset_v3.csv  # Multimodal metadata (without impression)
│       │
│       ├── reports/txt/                         # Raw patient report files (.txt)
│       │
│       ├── Notebooks/
│       │   ├── Phase-1mimic_image_classifier.ipynb         # Phase 1 notebook
│       │   └── Phase-2-FINAL_multimodal_classifier.ipynb   # Phase 2 notebook
│       │
│       ├── outputs/                              # Phase 1 checkpoints & plots
│       │   ├── best_pneumofusion_mimic.pth       # Phase 1 model weights (102MB)
│       │   ├── image_features_train.pt           # Extracted image features (train)
│       │   ├── image_features_val.pt             # Extracted image features (val)
│       │   └── image_features_test.pt            # Extracted image features (test)
│       │
│       └── outputs_phase2/                       # Phase 2 checkpoints & plots
│           ├── best_phase2_multimodal.pth        # Phase 2 model weights (437MB)
│           ├── fused_features_train.pt           # Extracted 1792-d features (train)
│           ├── fused_features_val.pt             # Extracted 1792-d features (val)
│           ├── fused_features_test.pt            # Extracted 1792-d features (test)
│           ├── phase1_vs_phase2_comparison.png   # Performance comparison chart
│           ├── phase2_results.png                # Confusion matrix & ROC curve
│           └── phase2_training_history.png       # Training curves
│
├── README.md
└── .gitignore
```

---

## ⚙️ Technologies Used

- **Frameworks**: Python, PyTorch, PyTorch Lightning
- **Backbone models**: ResNet50 (pre-trained), Emily Alsentzer's Bio_ClinicalBERT
- **Libraries**: HuggingFace Transformers, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Hardware**: CUDA-enabled GPUs (NVIDIA RTX 3050 Laptop GPU)

---

## 👨‍💻 Author

### Ashutosh Yadav
- **Affiliation**: Indian Institute of Technology Guwahati (IIT Guwahati)
- **Specialization**: B.Sc. in Data Science & Artificial Intelligence
- **Email**: [ay346185@gmail.com](mailto:ay346185@gmail.com)
