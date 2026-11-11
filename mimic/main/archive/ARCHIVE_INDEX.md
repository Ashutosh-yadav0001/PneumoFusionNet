# PneumoFusionNet Archive Index

This directory archives legacy experiments, superseded notebook versions, and previous training runs conducted prior to the standardized **Experiments 1–6 Pipeline** defined by the research supervisor.

---

## 📁 Archive Structure

### `notebooks/`
* **`legacy_phase_1/`**: Early single-view and cross-validation iterations (v1.1 through v1.2, v4-v6, balanced/scaleup experiments).
* **`legacy_phase_2/`**: Early multimodal fusion notebooks (Phase-2-multimodal_fusion_PA.ipynb).
* **`legacy_phase_3/`**: Prior triple-fusion implementations (vitals, 11-feature, scaleup, dual clinical/image experiments).
* **`legacy_phase_4/`**: Dual fusion exploratory notebooks (Phase-4a image+WBC).
* **`legacy_scaleup/`**: Intermediate scaleup test notebooks.
* **`legacy_main_subset/`**: Initial 1,000-sample pilot subset notebooks.
* **`exploratory_exp2_checkpoints/`**: Exploratory draft notebooks comparing CheXpert vs NIH backbones.
* **`dataset_build_scripts/`**: One-time ETL scripts used to download, map, and prepare the MIMIC-CXR subset.

### `outputs/`
* **`legacy_phase_1_outputs/`**: Checkpoints and evaluation plots from old Phase 1 runs (including runs using the contaminated `densenet121-res224-all` backbone).
* **`legacy_phase_2_outputs/`**: Checkpoints and evaluation plots from early Phase 2 multimodal experiments.
* **`legacy_phase_3_outputs/`**: Checkpoints and feature vectors from prior Phase 3 runs.
* **`legacy_phase_4_outputs/`**: Phase 4 exploratory dual-fusion outputs.
* **`exploratory_exp2/`**: Outputs from preliminary checkpoint overlap tests.

---

## 🎯 Active Project Structure (`mimic/main/`)

All current and future experiments adhere strictly to the teacher's guidelines:
* **Active Notebooks**: Located in `mimic/main/Experiment_1_Notebooks/`
  1. `Phase-1_Image_Only_Exp1.ipynb` — Experiment 2 & 3: Clean NIH DenseNet-121 baseline with validation-derived threshold.
  2. `Phase-2_Multimodal_Image_Text_Exp1.ipynb` — Experiment 4: Multimodal (CXR + Report Text) + Text-Only baseline.
  3. `Phase-3_Triple_Fusion_Exp1.ipynb` — Experiment 5 & 6: 4-way comparison and clean clinical feature fusion without leakage.
* **Active Outputs**: Located in `mimic/main/outputs/`
  - `Phase_1_NIH_single_split_exp2/` — Final clean Phase 1 baseline model and metrics.
  - `Phase_2_NIH_single_split_exp2/` — Final clean Phase 2 multimodal model.
* **Canonical Split**: `mimic/main/dataset/patient_split.csv` (Patient-level 70/15/15 stratified partition, seed=42, 3,404 subjects, zero patient overlap).
