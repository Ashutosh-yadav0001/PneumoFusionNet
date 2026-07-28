# Changelog

All notable changes to PneumoFusionNet are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]
- Phase 3 integration with full MIMIC-IV admission-level clinical features
- Grad-CAM visualisation utility in `src/utils/`
- Automated test suite (`tests/`)

---

## [1.0.0] — 2026-07-28

### Added
- Modular `src/` Python package extracted from all experiment notebooks:
  - `src/models/vision.py` — `ChannelAttention`, `SpatialAttention`, `CBAM`, `EnhancedPneumoNetV4`, `ImageEncoder`
  - `src/models/text_encoder.py` — `TextEncoder` (Bio_ClinicalBERT partial fine-tuning)
  - `src/models/fusion.py` — `CrossAttnFusionNet`, `MetadataEncoder`, `TripleFusionNet`
  - `src/data/dataset.py` — `CXRDataset`, `MultimodalCXRDataset`, `TripleModalCXRDataset`
  - `src/data/preprocessing.py` — CLAHE, lung bbox detection, anti-leakage text builder, TTA transforms
  - `src/utils/metrics.py` — evaluation, TTA, threshold selection, permutation importance
  - `src/utils/training.py` — `FocalLoss`, Mixup variants, training loop, optimizer builder
- Added `Phase-3c-wbc_only_fusion.ipynb` and `Phase-3-triple_fusion_PA_WBC_ONLY_scaleup_features.ipynb` for WBC-specific clinical fusion experiments.
- `pyproject.toml` — project is now `pip install -e .` installable
- `LICENSE` (MIT)
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `CHANGELOG.md`

---

## [0.9.0] — 2026-07 (Phase 3 Triple Fusion — SOTA)

### Added
- **Phase 3 Triple Fusion** on ~3,763 PA-view images:
  - Metadata MLP: 17 clinical features (demographics + vitals + labs + `has_vitals` flag)
  - Concatenates `[img(1024) + cross_attn(512) + meta(64)] = 1600-d`
  - Warm-started cross-attention from Phase 2v2 checkpoint
  - Triple Mixup across all three embedding branches
  - Permutation feature importance for all 17 clinical features

### Results
| Dataset   | AUC    | Sensitivity | Specificity | Accuracy |
|:----------|:------:|:-----------:|:-----------:|:--------:|
| Half (~1,857) | 0.9841 | 94.9% | 94.5% | 94.7% |
| Scaleup (~3,763) | **0.9890** | 89.1% | 94.3% | **91.7%** |

---

## [0.8.0] — 2026-06 (Phase 2v2 — Publication-Grade SOTA)

### Added
- **Phase 2v2 Cross-Attention Fusion** (image + text):
  - Switched from frozen ClinicalBERT → unfroze last 2 layers (layers 10 & 11)
  - Expanded input text: `FINDINGS` only → `FINDINGS + HISTORY`
  - Replaced simple concatenation fusion → 8-head MultiheadCrossAttention
  - Replaced CrossEntropyLoss → Focal Loss (γ=2.0)
  - Youden-J + clinical threshold (≥90% sensitivity) evaluation

### Results
| Metric | Phase 2v1 | Phase 2v2 | Δ |
|:-------|:---------:|:---------:|:---:|
| AUC | 0.9109 | **0.9490** | +3.81 |
| Sensitivity | 80.6% | **91.37%** | +10.77% |
| Accuracy | 85.3% | **88.63%** | +3.33% |

---

## [0.7.0] — 2026-05 (Phase 2v1 — First Multimodal)

### Added
- Phase 2 multimodal pipeline (image + text):
  - Frozen Bio_ClinicalBERT encoder on `FINDINGS` section only
  - Simple concatenation fusion (image + CLS token)
  - Anti-leakage protocol: IMPRESSION section excluded, diagnostic keywords redacted
- `MultimodalCXRDataset` with tokenizer integration
- `mimic_paired_dataset_phase2.csv` (image–report manifest)

### Results
- AUC: **0.9109** (+6.6% over Phase 1.1v5 image-only)

---

## [0.6.0] — 2026-04 (Phase 1.1v5 — Peak Image-Only)

### Added
- Phase 1.1v5: DenseNet-121 + CBAM + Batch Size 16 + Youden-J threshold
- Peak image-only ceiling: AUC **0.8591**, Sensitivity 76.4%, Specificity 79.3%

---

## [0.5.0] — 2026-03 (Phase 1.1v4 — Cross-Validation)

### Added
- Phase 1.1v4: 5-Fold GroupKFold cross-validation (zero patient leakage)
- Test-Time Augmentation (TTA) with 10 medically-safe augmentation views
- Lung bounding-box pre-computation via Otsu + contour detection
- CLAHE preprocessing integrated into `EnhancedCXRDataset`

### Results
- AUC: **0.8445**, Sensitivity 79.5%, Specificity 76.0%

---

## [0.4.0] — 2026-02 (Phase 1.1 Scale-Up)

### Added
- Scale-up from 139-image pilot to ~1,989 PA-view images
- ResNet50 + GCSA (Global Context Spatial Attention) pilot architecture

### Results
- AUC: **0.7126** (raw unoptimised scale-up baseline)

---

## [0.1.0] — 2026-01 (Phase 1 Pilot Baseline)

### Added
- Initial 139-image MIMIC-CXR pilot cohort
- Custom ResNet50 + GCSA architecture
- Binary pneumonia classification baseline
- AUC: **0.6667** on pilot cohort
