# PneumoFusionNet: Pilot Experiments Summary & Scale-Up Roadmap

This document serves as the final report for the **PneumoFusionNet Pilot Study** using the 139-sample cohort. It outlines the methodologies, results, key insights on statistical variance, and establishes the blueprint for the next phase ($1000+$ samples).

---

## 🎯 Project Overview & Objective
The goal of **PneumoFusionNet** is to develop a robust multimodal fusion network that classifies chest radiographs for the presence of **Pneumonia** by combining:
1. **Visual Modality:** Frontal Chest X-rays (CXR).
2. **Textual Modality:** Semantically parsed clinical history and indications from radiology reports.
3. **Structured Modality (Phase 3):** Physiological lab measurements (WBC, CRP, Procalcitonin).

The pilot study was designed to establish a baseline, validate the fusion pipeline, and identify architectural limitations before scaling to a larger cohort.

---

## 📊 Performance Comparison: All Pilot Phases

The model was evaluated across 4 distinct phases on the test split (using `SEED=42`):

| Phase | Modality | Dataset | Test AUC | Test Accuracy | Test F1 (Macro) | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 1** | Image Only | Imbalanced (118:21) | 0.6667 | 0.7619 | 0.4300 | Completed |
| **Phase 1.1** | Image Only | **Balanced (21:21)** | **0.9167** | **0.8571** | **0.8571** | Completed |
| **Phase 2** | Image + Text | Imbalanced (118:21) | 0.7037 | 0.8095 | 0.4474 | Completed |
| **Phase 2.1** | Image + Text | **Balanced (21:21)** | **0.9167** | **0.7143** | **0.7083** | Completed |

---

## 📐 Understanding the Accuracy Swing (Phase 1.1 vs. Phase 2.1)

In the balanced subset experiments (Phases 1.1 and 2.1), the cohort consists of **42 total samples**. Using a standard $70/15/15$ split:
* **Train Set:** 29 samples
* **Val Set:** 6 samples
* **Test Set:** **7 samples**

Because the test set has only $7$ samples, any change in prediction on a single sample causes a massive **$14.28\%$ swing** in accuracy:
* **Phase 1.1 Accuracy (85.71%)** = $6$ out of $7$ correct.
* **Phase 2.1 Accuracy (71.43%)** = $5$ out of $7$ correct.

$$\Delta = 1 \text{ sample classification change}$$

### 🔬 Statistical Significance Proof
To verify whether the difference is meaningful, we compute the standard error ($SE$) of the sample proportions:

$$SE = \sqrt{\frac{p(1-p)}{N}}$$

* **Phase 1.1 CI ($95\%$):** $0.8571 \pm 0.2593 \approx [59.78\%, 100\%]$
* **Phase 2.1 CI ($95\%$):** $0.7143 \pm 0.3346 \approx [37.97\%, 100\%]$

The confidence intervals overlap almost completely, and a **Fisher's Exact Test** yields a **$p$-value $\approx 1.000$**. This confirms that the $14.28\%$ difference is **statistically insignificant** and is purely random variance caused by the small sample size ($N=7$).

---

## 🔑 Critical Observations from the Pilot

1. **Projection Mismatch (AP/PA/Lateral):** 
   The current dataset includes all available projections per study. PA, AP, and Lateral views look dramatically different, creating unnecessary visual noise that confuses the visual encoder.
2. **Text Input & Target Leakage:**
   Radio-logical reports contain the final diagnosis in the `IMPRESSION` section. Stripping this section ensures the model learns predictive cues from pre-diagnosis text (e.g. `INDICATION`), but makes the text signal highly sparse in very small cohorts.
3. **Patient-Level Leakage:**
   A standard row-level split can partition multiple studies from the same patient across train/val/test splits, leaking anatomy and artificially boosting validation metrics.

---

## 🚀 Roadmap for the Next Phase ($1000+$ Samples)

To transition to a robust, publication-grade model, the next phase will implement the following changes:

### 1. Dataset Expansion & Curation
* **Scale up to $1000+$ samples** (500 Normal, 500 Pneumonia) to reduce evaluation margins of error below $\pm 3\%$.
* **Standardize Projections:** Filter the dataset to use **PA-only (Posteroanterior)** frontal chest X-rays. Exclude AP and Lateral views.
* **Deduplication:** Limit the dataset to a maximum of one image/study per patient.

### 2. Grouped Splitting
* Implement `GroupShuffleSplit` or `StratifiedGroupKFold` grouped by patient ID (`subject_id`) to ensure complete patient isolation across splits.

### 3. Modality Expansion (Phase 3: MIMIC-IV Clinical Labs)
* Extract and match patient clinical lab values (White Blood Cell Count, C-Reactive Protein, and Procalcitonin) corresponding to the time of the radiograph.
* Fuse: **Image (1024-d) + Text (768-d) + Labs (N-d)** into a unified classification MLP.
