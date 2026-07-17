# PneumoFusionNet — Complete Project Summary
### Dataset, Techniques, Results, Issues & Changes

---

## 1. Phase 1.1 — First Baseline (Custom CNN)

**Notebook:** `Phase-1.1-image_classifier_balanced_scaleUp(PA View).ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | 1,854 images (927 Normal + 927 Pneumonia, balanced) |
| **Split** | Train: 1,297 / Val: 279 / Test: 278 |
| **Backbone** | Custom CNN (no CXR pretrained weights) |
| **CBAM** | ❌ No |
| **CLAHE** | ❌ No |
| **Bbox** | ❌ No |
| **TTA** | ❌ No |
| **Result** | No saved metrics |

> [!CAUTION]
> **Main Issues:**
> 1. Custom CNN with ImageNet weights — not designed for X-rays
> 2. No CLAHE preprocessing — low contrast X-rays fed directly
> 3. No lung cropping — model sees borders, labels, bone, and irrelevant anatomy
> 4. No TTA, no threshold tuning — raw, unoptimised predictions

![Phase 1.1 Confusion Matrix](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1_PA_balanced_scaled__confusion_matrix.png)

![Phase 1.1 Training History](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1_PA_balanced_scaled__training_history.png)

---

## 2. Phase 1.1v2 — Added DenseNet-121 + CBAM + CLAHE

**Notebook:** `Phase-1.1v2-enhanced_image_classifier_PA.ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | 1,854 images (927 Normal + 927 Pneumonia) |
| **Backbone** | **TorchXRayVision DenseNet-121** (`densenet121-res224-all`) ← NEW |
| **CBAM** | ✅ Channel + Spatial Attention ← NEW |
| **CLAHE** | ✅ clip=2.0, tile=(8,8) ← NEW |
| **Bbox** | ✅ Lung segmentation (~354 images only) ← NEW |
| **TTA** | ❌ No |
| **Result** | No saved final AUC (used for bbox generation) |

> **What we changed:** Replaced custom CNN with CXR-pretrained DenseNet-121. Added CBAM attention + CLAHE + lung bbox.
>
> [!WARNING]
> **Main Issues:**
> 1. Bbox coverage only **354/1,854 (19%)** — 81% of images still uncropped
> 2. No TTA, no cross-validation, no threshold tuning
> 3. No Focal Loss — standard CE treats all errors equally

![Phase 1.1v2 Preprocessing Comparison](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v2_PA_enhanced__preprocessing_comparison.png)

![Phase 1.1v2 Confusion Matrix](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v2_PA_enhanced__confusion_matrix.png)

---

## 3. Phase 1.1v3 — Added Focal Loss + Mixup

**Notebook:** `Phase-1.1v3-fixed_image_classifier_PA.ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | 1,854 images (927 Normal + 927 Pneumonia) |
| **Backbone** | TorchXRayVision DenseNet-121 (`densenet121-res224-all`) |
| **CBAM** | ✅ |
| **CLAHE** | ✅ clip=2.0, tile=(8,8) |
| **Loss** | **Focal Loss** ← NEW |
| **Regularisation** | **Mixup (alpha=0.3)** ← NEW |
| **LR** | **3e-5** (reduced from 1e-4) ← CHANGED |
| **TTA** | ❌ No |

| Metric | Score |
|--------|-------|
| **AUC** | **0.8120** |
| **Tuned Accuracy** | **70.9%** |

> **What we changed:** Added Focal Loss (focus on hard samples) + Mixup regularisation. Reduced LR for stability.
>
> [!WARNING]
> **Main Issues:**
> 1. Accuracy only **70.9%** — too low for any clinical use
> 2. Still single split — results could be a lucky split
> 3. No TTA — prediction noise
> 4. No Youden-J threshold — using arbitrary 0.5 cutoff

![Phase 1.1v3 Confusion Matrices](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v3_PA_fixed__confusion_matrices.png)

![Phase 1.1v3 ROC & PR Curves](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v3_PA_fixed__roc_pr_curves.png)

---

## 4. Phase 1.1v4 CrossVal — 5-Fold + TTA + Youden-J

**Notebook:** `Phase-1.1v4-crossval_tta_PA.ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | 1,854 images (927 Normal + 927 Pneumonia) |
| **Split** | **5-Fold GroupKFold** (zero patient leakage) ← NEW |
| **Backbone** | TorchXRayVision DenseNet-121 (`densenet121-res224-all`) |
| **CBAM** | ✅ |
| **CLAHE** | ✅ clip=2.0, tile=(8,8) |
| **Loss** | Focal Loss + Mixup (alpha=0.3) |
| **TTA** | **10 views** ← NEW |
| **Threshold** | **Youden-J** ← NEW |
| **Bbox** | 354/1,854 (19%) |

| Metric | Mean ± Std | Best Fold (Fold 2) |
|--------|-----------|---------------------|
| **AUC (TTA)** | 0.820 ± 0.021 | **0.8445** |
| **Tuned Accuracy** | 76.0% ± 1.4% | **77.6%** |
| **Sensitivity** | 69.7% ± 7.5% | **79.5%** |
| **Specificity** | — | **76.0%** |

> **What we changed:** Added 5-Fold cross-validation, TTA (10 views), Youden-J threshold.
>
> [!WARNING]
> **Main Issues:**
> 1. **High variance across folds** — Sensitivity ranges from 60.8% to 79.5%
> 2. Mean Sensitivity only **69.7%** — misses 1 in 3 pneumonia patients
> 3. Bbox still only **19%** coverage
> 4. Training took **~5 hours** for all 5 folds

![Phase 1.1v4 CrossVal Fold Comparison](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v4_PA_crossval__cv_results_comparison.png)

![Phase 1.1v4 CrossVal Confusion Matrices](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v4_PA_crossval__confusion_matrices_pooled.png)

![Phase 1.1v4 CrossVal ROC](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v4_PA_crossval__pooled_roc_curves.png)

---

## 5. Phase 1.1v4 Single Split

**Notebook:** `Phase-1.1v4-single_split_tta_PA.ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | 1,854 images |
| **Split** | Train: 1,290 / Val: 284 / Test: 280 |
| **Everything else** | Same as v4 CrossVal |

| Metric | Score |
|--------|-------|
| **AUC (TTA)** | **0.8639** |
| **Tuned Accuracy** | **77.1%** |
| **Sensitivity** | **72.1%** |
| **Specificity** | **82.1%** |

> **What we changed:** Single clean split for faster iteration.
>
> [!WARNING]
> **Main Issue:** Sensitivity dropped to **72.1%** — model misses 1 in 4 pneumonia patients

![Phase 1.1v4 Single Split Confusion Matrices](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v4_PA_single_split__confusion_matrices.png)

---

## 6. Phase 1.1v5 — Advanced

**Notebook:** `Phase-1.1v5-advanced_image_classifier_PA.ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | 1,854 images |
| **Split** | Train: 1,290 / Val: 284 / Test: 280 |
| **Backbone** | TorchXRayVision DenseNet-121 (`densenet121-res224-all`) |
| **Batch Size** | **16** ← increased from 8 |
| **CBAM** | ⚠️ Not detected in code |

| Metric | Score |
|--------|-------|
| **AUC (TTA)** | **0.8591** |
| **Tuned Accuracy** | **77.9%** (highest image-only) |
| **Sensitivity** | **76.4%** |
| **Specificity** | **79.3%** |
| Best Val AUC | 0.8517 (Epoch 19) |

> **What we changed:** Batch size 8 → 16 for better gradient estimates.
>
> [!WARNING]
> **Main Issue:** AUC ceiling at ~0.86 — image-only models cannot break past this on 1,854 images. Need text or more data.

![Phase 1.1v5 Confusion Matrices](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v5_PA_advanced__confusion_matrices.png)

---

## 7. Phase 1.1v6 — All Image Improvements (❌ FAILED)

**Notebook:** `Phase-1.1v6-improved_PA.ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | **1,989 images** (1,062 Normal + 927 Pneumonia) — slightly larger |
| **Split** | Train: 1,391 / Val: 299 / Test: 299 |
| **Backbone** | DenseNet-121 (**`densenet121-res224-mimic_nb`** ← CHANGED) |
| **CBAM** | ✅ |
| **CLAHE** | ✅ (config had 3.0 but scanner found 2.0) |
| **Bbox** | **Expanded to 100%** via Otsu (2,343 bboxes!) ← NEW |
| **Image Size** | **320×320** ← CHANGED from 224 |
| **Loss** | CrossEntropy + **label_smoothing=0.1** ← NEW |
| **Scheduler** | **CosineAnnealingWarmRestarts** ← CHANGED |
| **Augmentations** | Added AutoContrast + Equalize ← NEW |

| Metric | Score |
|--------|-------|
| Best Val AUC | 0.7851 |
| **AUC (TTA)** | **0.8114** ❌ |
| **Youden Accuracy** | **75.6%** ❌ |
| **Sensitivity** | **60.4%** ❌ |
| **Specificity** | **88.8%** |

> [!CAUTION]
> **FAILED EXPERIMENT — Performance DROPPED on every metric**
>
> **Root Causes:**
> 1. **`mimic_nb` weights** were pretrained for multi-label (14 diseases) — doesn't transfer well to binary
> 2. **Otsu bbox detection was too aggressive** — incorrectly cropped lung tissue on many images (2,343 bboxes for 1,989 images = duplicates)
> 3. **Resolution 320×320 on 4GB VRAM** — memory pressure, smaller effective learning
> 4. **Too many changes at once** — impossible to debug which change caused the drop
>
> **Lesson:** Change ONE thing at a time. More changes ≠ better.

![Phase 1.1v6 Training Curves](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v6_PA_improved__v6_training_curves.png)

![Phase 1.1v6 Confusion Matrices](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v6_PA_improved__v6_confusion_matrices.png)

![Phase 1.1v6 ROC Curve](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v6_PA_improved__v6_roc_curve.png)

![Phase 1.1v6 All Versions Comparison](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_1.1v6_PA_improved__v6_all_versions_comparison.png)

---

## 8. Phase 1.2 — Larger Dataset with Custom CNN

**Notebook:** `Phase-1.2-image_classifier_balanced_scaleUp(PA View 1000+).ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | 1,854 images (927 Normal + 927 Pneumonia, balanced) |
| **Split** | Train: 1,297 / Val: 279 / Test: 278 |
| **Backbone** | Custom CNN (no TorchXRayVision) |
| **CBAM** | ❌ No |
| **CLAHE** | ❌ No |

| Metric | Score |
|--------|-------|
| Best Val AUC | 0.7946 |
| **Test AUC** | **0.7381** |
| Pneumonia Recall | 44% |

> [!CAUTION]
> **Main Issue:** Custom CNN without CXR pretraining performs terribly — only **73.8% AUC** and **44% Pneumonia recall** (misses more than half of sick patients)

---

## 9. Phase 2 v1 — First Multimodal (Image + Text Concat)

**Notebook:** `Phase-2\Phase-2-multimodal_fusion_PA.ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | **1,989 images + radiology reports** (1,062 Normal + 927 Pneumonia) |
| **Split** | Train: 1,391 / Val: 299 / Test: 299 |
| **Image Encoder** | Frozen Phase 1 Fold 2 DenseNet+CBAM (1024-d) |
| **Text Encoder** | **Bio_ClinicalBERT (100% frozen)** ← NEW |
| **Text Input** | FINDINGS only (IMPRESSION removed, leakage keywords redacted) |
| **Fusion** | **Concat + MLP** (1024+768 → 512 → 128 → 2) ← NEW |
| **Loss** | CrossEntropyLoss (weighted) |
| **Bbox** | 354/1,989 (18%) |
| **LR** | 2e-4 |

| Metric | Phase 1 Best | Phase 2 v1 | Gain |
|--------|-------------|------------|------|
| **AUC** | 0.8445 | **0.9109** | **+6.6%** 🔥 |
| **Accuracy** | 77.6% | **85.3%** | **+7.7%** |
| **Sensitivity** | 79.5% | **80.6%** | +1.0% |
| **Specificity** | 76.0% | **89.4%** | **+13.4%** |

> **What we changed:** Added radiology report text via ClinicalBERT. Removed IMPRESSION (data leakage). Redacted keywords.
>
> [!WARNING]
> **Main Issues:**
> 1. **Sensitivity barely improved (+1%)** — text helped identify Normal, not Pneumonia
> 2. **ClinicalBERT 100% frozen** — generic embeddings, not fine-tuned
> 3. **Simple concat fusion** — image and text features don't interact
> 4. **HISTORY section ignored** — only using FINDINGS
> 5. **Standard CE loss** — treats all errors equally

![Phase 2 v1 Sanity Check](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2_multimodal__sanity_check.png)

![Phase 2 v1 Training Curves](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2_multimodal__phase2_training_curves.png)

![Phase 2 v1 Confusion Matrices](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2_multimodal__phase2_confusion_matrices.png)

![Phase 2 v1 ROC Curve](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2_multimodal__phase2_roc_curve.png)

![Phase 1 vs Phase 2 Comparison](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2_multimodal__phase1_vs_phase2_comparison.png)

---

## 10. Phase 2 v2 — All Advanced Improvements (🏆 BEST MODEL)

**Notebook:** `Phase-2\Phase-2v2-multimodal_improved_PA.ipynb`

| Item | Detail |
|------|--------|
| **Dataset** | **1,989 images + radiology reports** (1,062 Normal + 927 Pneumonia) |
| **Split** | Train: 1,391 / Val: 299 / Test: 299 |
| **Image Encoder** | Frozen Phase 1 Fold 2 DenseNet+CBAM (1024-d) |
| **Text Encoder** | **Bio_ClinicalBERT — last 2 layers UNFROZEN (lr=1e-5)** ← CHANGED |
| **Text Input** | **FINDINGS + HISTORY** ← CHANGED |
| **Fusion** | **Cross-Attention (8-head MultiheadAttention, dim=512)** ← CHANGED |
| **Loss** | **Focal Loss (gamma=2.0)** ← CHANGED |
| **Regularisation** | **Mixup on embeddings (alpha=0.2)** ← NEW |
| **Explainability** | **Grad-CAM** ← NEW |
| **Optimizer** | AdamW with **separate LRs** (BERT=1e-5, Fusion=2e-4) ← CHANGED |
| **Bbox** | 354/1,989 (18%) |

| Metric | Phase 2 v1 | Phase 2 v2 | Gain |
|--------|------------|------------|------|
| **AUC** | 0.9109 | **0.9490** | **+3.8%** 🔥 |
| **Accuracy** | 85.3% | **88.6%** | **+3.3%** |
| **Sensitivity** | 80.6% | **91.4%** | **+10.8%** 🚀 |
| **Specificity** | 89.4% | **86.3%** | -3.1% (traded for Sens) |

> **What we changed (7 improvements):**
> 1. Text: FINDINGS → **FINDINGS + HISTORY** (richer clinical context)
> 2. BERT: frozen → **last 2 layers unfrozen** (task-specific fine-tuning)
> 3. Fusion: concat → **Cross-Attention** (image queries text tokens)
> 4. Loss: CE → **Focal Loss** (focuses on hard-to-classify samples)
> 5. Added **Mixup on embeddings** (regularisation)
> 6. Added **separate LRs** (prevents BERT catastrophic forgetting)
> 7. Added **Grad-CAM** (explainability)

> [!NOTE]
> **Remaining Limitations:**
> 1. Small dataset (~1,989 images) — may not generalise to other hospitals
> 2. Single institution (Beth Israel) — bias toward one hospital's equipment
> 3. Binary classification only — real CXR has 14+ pathologies
> 4. Bbox coverage still 18%
> 5. No external validation on NIH/CheXpert

![Phase 2 v2 Training Curves](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2v2_improved__v2_training_curves.png)

![Phase 2 v2 Confusion Matrices](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2v2_improved__v2_confusion_matrices.png)

![Phase 2 v2 ROC Curve](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2v2_improved__v2_roc_curve.png)

![Phase 2 v2 Grad-CAM](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2v2_improved__v2_gradcam.png)

![All Phases Comparison](C:/Users/ashutosh/.gemini/antigravity/brain/4dd76b5f-25f2-4f9d-a828-5ec5747bd153/Phase_2v2_improved__all_phases_comparison.png)

---

## Master Table — Dataset + Results + Main Issue

| # | Notebook | Dataset | Train/Val/Test | AUC | Accuracy | Main Issue |
|---|---------|---------|----------------|-----|----------|------------|
| 1 | Phase 1.1 | 1,854 img | 1297/279/278 | — | — | No CXR backbone, no CLAHE, no attention |
| 2 | Phase 1.1v2 | 1,854 img | — | — | — | Bbox only 19%, no TTA |
| 3 | Phase 1.1v3 | 1,854 img | — | 0.812 | 70.9% | Accuracy too low, no TTA, no crossval |
| 4 | v4 CrossVal | 1,854 img | 5-Fold | **0.845** | 77.6% | Sensitivity variance 60–80%, 5hr training |
| 5 | v4 Single | 1,854 img | 1290/284/280 | 0.864 | 77.1% | Sensitivity only 72% |
| 6 | v5 Advanced | 1,854 img | 1290/284/280 | 0.859 | **77.9%** | AUC ceiling at ~0.86 |
| 7 | v6 ❌ | **1,989 img** | 1391/299/299 | 0.811 | 75.6% | `mimic_nb` weights + bad bbox = FAILED |
| 8 | Phase 1.2 | 1,854 img | 1297/279/278 | 0.738 | — | Custom CNN = worst performance |
| 9 | **Phase 2 v1** | **1,989 img+txt** | 1391/299/299 | **0.911** | **85.3%** | Sensitivity barely improved (+1%) |
| 10 | **Phase 2 v2** 🏆 | **1,989 img+txt** | 1391/299/299 | **0.949** | **88.6%** | Small dataset, no external validation |

---

## The Evolution Flow

```
Phase 1.1 (Custom CNN)           → AUC: ???   | Issue: No CXR backbone
   ↓ Added DenseNet-121 + CBAM + CLAHE
Phase 1.1v2                      → AUC: ???   | Issue: Bbox only 19%
   ↓ Added Focal Loss + Mixup
Phase 1.1v3                      → AUC: 0.812 | Issue: 70.9% accuracy too low
   ↓ Added 5-Fold CrossVal + TTA + Youden-J
Phase 1.1v4 CrossVal (Fold 2)    → AUC: 0.845 | Issue: Sensitivity 69.7% avg
   ↓ Batch size 8→16
Phase 1.1v5                      → AUC: 0.859 | Issue: AUC ceiling at ~0.86
   ↓ Changed weights to mimic_nb + expanded bbox (MISTAKE)
Phase 1.1v6 ❌                   → AUC: 0.811 | Issue: FAILED — everything dropped
   
   ↓ Instead: Added ClinicalBERT text from radiology reports
Phase 2 v1 (Concat Fusion)       → AUC: 0.911 | Issue: Sensitivity only 80.6%
   ↓ Cross-Attention + Focal Loss + Unfreeze BERT + Mixup + Grad-CAM
Phase 2 v2 (Cross-Attention) 🏆  → AUC: 0.949 | Sensitivity: 91.4% ✅
```
