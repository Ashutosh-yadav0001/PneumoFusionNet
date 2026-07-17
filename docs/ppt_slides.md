# 🎞️ PneumoFusionNet — Part 1 PPT Slide Content
### For Google Slides / PowerPoint | 6–8 min | 15 Slides

> **Design System:** Dark navy background `#0D1B2A`, cyan accent `#00E5FF`,
> orange for warnings `#E67E22`, green for results `#2ECC71`, red for failures `#E74C3C`.
> Font: **Montserrat Bold** headings (32–40pt white), **Roboto** body (18pt `#CBD5E0`).
> Key numbers: 56pt bold white on teal card `#1A5276`.

---

## SLIDE 1 — Title Slide

```
┌──────────────────────────────────────────────────────────────┐
│  [Background: Blurred chest X-ray + neural network overlay]  │
│                                                              │
│  🫁  PneumoFusionNet                                         │
│      Part 1: Data Preparation & Pilot Experiments           │
│                                                              │
│      Explainable Multimodal Deep Learning                   │
│      for Pneumonia Detection                                 │
│      from MIMIC-CXR X-Rays & Clinical Reports               │
│                                                              │
│  ──────────────────────────────────────────────────────────  │
│  Ashutosh Yadav  |  Roll: 23035010693                       │
│  B.Sc. (Hons.) Data Science & AI — IIT Guwahati            │
│  Trimester 8 Term Project  |  July 2026                     │
└──────────────────────────────────────────────────────────────┘
```

**Speaker note:** "Today I'll walk you through Part 1 of PneumoFusionNet — covering the full 4-stage experimental journey from public Kaggle/IU dataset architecture tests, to the 139-sample MIMIC pilot, to 1,989-sample scale-up (image-only), and finally multimodal fusion using restricted clinical data."

---

## SLIDE 2 — The Clinical Problem & Our Solution

**Heading:** Why Pneumonia Diagnosis Needs AI + Clinical Text Together

**Left 55% — 3 problem cards:**

```
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│ 🫁 VISUAL AMBIGUITY  │  │ 🔑 LABEL LEAKAGE      │  │ 📊 DATA REALITY      │
│                      │  │                      │  │                      │
│ Consolidation =      │  │ 90%+ of multimodal   │  │ MIMIC-CXR is         │
│ Pneumonia OR         │  │ AI models read the   │  │ RESTRICTED — requires│
│ Atelectasis OR       │  │ diagnosis from the   │  │ CITI training + DUA  │
│ Effusion?            │  │ IMPRESSION section   │  │ (not publicly        │
│                      │  │ — that's cheating!   │  │ downloadable)        │
└──────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

**Right 45% — Solution pyramid:**

```
  🏆 Part 1 Goal:
  ┌────────────────────────────────────────┐
  │  Stage 3: MIMIC Multimodal Fusion      │  Image + Anti-Leakage Text
  │  Stage 2: MIMIC Scale-Up (1,989 PA)    │  Image-Only Architecture Race
  │  Stage 1: MIMIC Pilot (139 samples)    │  First Real Clinical Data
  │  Stage 0: Public Dataset (Kaggle/IU)   │  Architecture Validation
  └────────────────────────────────────────┘
```

---

## SLIDE 3 — Stage 0: Public Dataset Experiments

**Heading:** Stage 0 — Architecture Validation (Kaggle / IU Chest X-Ray)

**Dataset box (left 40%):**

```
┌────────────────────────────────────────┐
│  📦 PUBLIC DATASETS USED               │
│                                        │
│  Kaggle: Chest X-Ray Pneumonia         │
│  Train: 5,216 | Test: 624              │
│  (Normal vs Pneumonia, balanced)       │
│                                        │
│  IU X-Ray (multimodal prototype)       │
│  (Indiana University, open access)     │
│                                        │
│  ⚠️ Pre-cleaned, balanced, easy        │
│  NOT representative of clinical data   │
└────────────────────────────────────────┘
```

**4 experiment result cards (right 60%):**

| Exp | Model | Dataset | AUC | Notes |
|-----|-------|---------|-----|-------|
| v1 | Basic ResNet50 | Kaggle | ~0.97 | ⚠️ Clean data |
| v2 | ResNet50 + DSC + GCSA | Kaggle | 92% Acc | ✅ Arch works |
| v3-1 | Image baseline | IU dataset | — | Multimodal prep |
| v3-2 | Image + Text fusion | IU dataset | — | Pipeline prototype |

**Bottom orange box:**
> ⚠️ 92% accuracy on Kaggle ≠ clinical readiness. These experiments only confirmed the architecture code works — not that it can handle real restricted hospital data.

---

## SLIDE 4 — Why MIMIC-CXR & The Anti-Leakage Design

**Heading:** Stage 1 — Entering MIMIC-CXR + Anti-Leakage Pipeline

**Left — MIMIC vs Public table:**

| Dimension | Kaggle/IU | MIMIC-CXR |
|-----------|-----------|-----------|
| Access | Open / Kaggle | **Restricted — CITI + DUA** |
| Balance | 50/50 | **5.6:1 imbalance** |
| Views | Single | PA + AP + Lateral mixed |
| Labels | Manual | CheXpert NLP auto-labels |
| Patients | Unknown | Multi-study per patient risk |

**Right — Anti-leakage pipeline:**

```
Raw MIMIC-CXR Report
─────────────────────────────────
INDICATION: 45M, 3-day fever       ✅ KEEP
FINDINGS: Opacity left lower lobe  ✅ KEEP
IMPRESSION: PNEUMONIA.             ❌ STRIPPED

        ↓ rebuild_dataset_findings.py
        ↓ Bio_ClinicalBERT tokenizer
        768-d anti-leakage text embedding
```

**Bottom callout (orange):**
> ⚠️ Without stripping IMPRESSION: model reads the answer — not a real diagnostic AI.

---

## SLIDE 5 — Stage 1: MIMIC Pilot (139 Samples) — 4 Phases

**Heading:** Stage 1 — 139-Sample Pilot: 4 Phases, Honest Results

**Dataset info bar:**
```
Pilot dataset: 139 samples (118 Normal + 21 Pneumonia = 5.6:1 imbalance)
Split: 97 train / 21 val / 21 test  |  Backbone: ResNet50 + DSC + GCSA
```

**4-phase results table:**

| Phase | Modality | Dataset | AUC | Accuracy | F1 (macro) |
|-------|----------|---------|-----|----------|------------|
| Ph. 1 | Image only | Imbalanced (118:21) | 0.667 🔴 | 76.2% | 0.430 |
| Ph. 1.1 | Image only | Balanced (21:21) | **0.917** ⚠️ | **85.7%** | 0.857 |
| Ph. 2 | Image + Text | Imbalanced (118:21) | **0.704** ✅ | **81.0%** | 0.447 |
| Ph. 2.1 | Image + Text | Balanced (21:21) | **0.917** ⚠️ | 71.4% | 0.708 |

**Multimodal gain highlight:**
> 📌 Text fusion on imbalanced data → **+3.7% AUC, +4.8% Accuracy** — 100% leakage-free

**Statistical warning box (orange):**
```
⚠️ Balanced pilot test set = 7 SAMPLES ONLY
   1 wrong prediction = 14.28% swing
   Fisher's Exact Test p ≈ 1.000
   → Statistically unreliable — scale-up required
```

---

## SLIDE 6 — Stage 2 Dataset: Scale-Up to 1,989 PA Images

**Heading:** Stage 2 — Scaling Up: From 139 Pilot → 1,989 PA-Only Images

**Left — Dataset evolution diagram:**

```
  PILOT (139)                 SCALE-UP (1,989)
  ──────────────              ──────────────────────────────
  All views                   PA-only (frontal, standardised)
  118 Normal + 21 Pneumonia   1,062 Normal + 927 Pneumonia
  Row-level splits            Patient-level GroupShuffleSplit
  ↑                           ↑
  View bias (AP vs PA)        Eliminated view confound
  Patient leakage risk        Zero patient overlap guaranteed

  Train / Val / Test
  Pilot:   97 / 21 / 21
  Scale:  1,401/ 295/ 293
```

**Right — Why PA-only matters:**

```
┌───────────────────────────────────────────────────┐
│  PA (Posteroanterior) vs AP (Anteroposterior)      │
│                                                   │
│  PA: Patient stands upright, beam from back       │
│      → standard diagnostic view, larger lungs    │
│                                                   │
│  AP: Bedridden patient, beam from front          │
│      → magnified heart, compressed lungs         │
│                                                   │
│  Mixing views = model learns VIEW TYPE not        │
│  lung pathology! PA-only eliminates this bias.   │
└───────────────────────────────────────────────────┘
```

**Bottom visual instruction:**
> 📊 **Chart to add:** Bar chart: 139 → 1,989 → showing Normal/Pneumonia breakdown

---

## SLIDE 7 — Stage 2: Architecture Race — Pilot Arch FAILS at Scale

**Heading:** ❌ Pilot Architecture (ResNet50 + DSC + GCSA) Fails on Real Clinical Data

**Side-by-side comparison:**

```
  PILOT (N=7 test — unreliable)    SCALE-UP (N=293 test — reliable)
  ──────────────────────────────   ─────────────────────────────────
  AUC:  0.917  ✅  (fake number)   AUC:  0.713  ❌  (real number)
  Acc:  85.7%  ✅  (on N=7)        Acc:  66.2%  ❌
  Sens: unreliable                  Sens: 69.9%  ❌ (misses 30% PN)
  Spec: unreliable                  Spec: 63.5%  ❌
```

**Root cause box (red):**
```
┌──────────────────────────────────────────────────────────────────┐
│  ❌ WHY ResNet50 + DSC + GCSA FAILED                             │
│                                                                  │
│  1. ImageNet pretraining — cats, dogs, cars. Never saw CXR.     │
│  2. 1,401 train images — too few to "un-learn" natural images   │
│  3. Domain shift — natural photos vs clinical X-ray texture     │
│                                                                  │
│  DECISION: Switch to DenseNet-121 with CXR-domain pretraining   │
└──────────────────────────────────────────────────────────────────┘
```

**Architecture switch box (cyan):**
```
FROM: ResNet50 (ImageNet) + DSC + GCSA
  TO: DenseNet-121 (torchxrayvision) + CBAM + CLAHE + Focal Loss

torchxrayvision DenseNet-121 pretrained on:
→ CheXpert     (224,316 CXR studies)
→ NIH ChestX-14 (112,120 CXR images)
→ MIMIC-CXR    (227,827 CXR images)
```

---

## SLIDE 8 — Stage 2: DenseNet-121 Evolution (8 Image-Only Experiments)

**Heading:** Stage 2 — DenseNet-121 Architecture Race (8 Iterations)

**Evolution timeline (horizontal):**

```
v1        v1.1v2         v1.1v3          v1.1v4          v1.1v5     v1.1v6❌
Custom →  DenseNet +  →  + Focal  →   + 5-Fold +  →   BS8→16  → mimic_nb +
CNN        CBAM+CLAHE     Loss+Mixup     TTA+Youden      →AUC0.86   Bbox100%
AUC:—      AUC: —         AUC: 0.812    AUC: 0.845       AUC:0.864  AUC:0.811❌
```

**Key results table:**

| Version | Key Change | AUC | Accuracy | Sensitivity | Issue |
|---------|------------|-----|----------|-------------|-------|
| v1.1 (Custom CNN) | No CXR backbone | — | — | — | Wrong arch |
| v1.1v3 | Focal Loss + Mixup | 0.812 | 70.9% | — | Too low |
| **v1.1v4 5-Fold** | CrossVal + TTA | **0.820±0.021** | 76.0% | 69.7% | Sensitivity variance |
| v1.1v5 | BatchSize 8→16 | 0.864 | 77.9% | 76.4% | AUC ceiling |
| **v1.1v6** | mimic_nb + Bbox 100% | **0.811 ❌** | **75.6%** | **60.4%** | **FAILED** |

**Lesson box (orange):**
> ⚠️ **v6 lesson:** Changed 4 things at once (wrong weights + aggressive bbox + bigger resolution + new scheduler) → performance crashed. **Change ONE variable at a time.**

---

## SLIDE 9 — Stage 2 Ceiling: Why Image-Only Can't Break AUC 0.87

**Heading:** Image-Only AUC Ceiling at ~0.86 → Need Clinical Text

**Left — AUC progression bar chart:**

```
  Phase 1.1v3:    ██████████░░░░  0.812
  Phase 1.1v4:    ███████████░░░  0.820±0.021
  Phase 1.1v5:    ████████████░░  0.864 ← ceiling
  Phase 1.1v6:    █████████████░  0.811 ← crashed
  Clinical target:░░░░░░░░░░░░░█  0.900
```

**Right — Why the ceiling exists:**

```
┌──────────────────────────────────────────────┐
│  What the image CANNOT tell you:             │
│                                              │
│  → Duration of symptoms (fever for 3 days?) │
│  → Lab results (WBC elevated?)               │
│  → Clinical context (immunocompromised?)     │
│  → Response to antibiotics?                  │
│                                              │
│  These are in the RADIOLOGY REPORT.          │
│  That's why we need multimodal fusion.       │
└──────────────────────────────────────────────┘
```

**Transition callout (cyan, bold):**
> 💡 Image-only max AUC ≈ 0.86. Adding clinical text (FINDINGS + HISTORY) → AUC 0.91 then 0.949.

---

## SLIDE 10 — Stage 3: Phase 2 v1 — First Multimodal Fusion

**Heading:** Stage 3 — Phase 2 v1: Multimodal Fusion (AUC 0.911)

**Architecture diagram (full width):**

```
  CXR Image (224×224)                  Radiology Report
       ↓                                      ↓
  DenseNet-121+CBAM (frozen)        Strip IMPRESSION
  from best Phase 1 fold (Fold 2)   ↓ FINDINGS only
       ↓                            Bio_ClinicalBERT (frozen)
  1024-d image embedding            768-d text embedding
       └──────────────┬─────────────────┘
                  Concat [1792-d]
                      ↓
                   MLP head → Normal / Pneumonia
```

**Result comparison:**

| Metric | Best Image-Only (v4 Fold2) | Phase 2 v1 | Gain |
|--------|--------------------------|------------|------|
| AUC | 0.845 | **0.911** | **+6.6%** 🔥 |
| Accuracy | 77.6% | **85.3%** | **+7.7%** |
| Sensitivity | 79.5% | **80.6%** | +1.0% |
| Specificity | 76.0% | **89.4%** | **+13.4%** |

**Issues box (orange):**
> ⚠️ Sensitivity barely improved (+1%) — text helps identify Normal, not Pneumonia.
> BERT fully frozen = generic embeddings. Simple concat = no cross-modal interaction.

---

## SLIDE 11 — Stage 3: Phase 2 v2 — Best Model (AUC 0.949) 🏆

**Heading:** Stage 3 — Phase 2 v2: 7 Improvements → AUC 0.949 🏆

**Left — 7 improvements:**

```
1. Text: FINDINGS → FINDINGS + HISTORY  (richer context)
2. BERT: frozen → last 2 layers unfrozen (task-specific tuning)
3. Fusion: Concat → Cross-Attention     (8-head, dim=512)
4. Loss: CE → Focal Loss (γ=2.0)       (hard sample focus)
5. Added Mixup on embeddings (α=0.2)    (regularisation)
6. Separate LRs: BERT=1e-5, Fusion=2e-4 (prevent forgetting)
7. Grad-CAM explainability              (which lung region?)
```

**Right — Final results:**

```
┌─────────────────────────────────────────────┐
│  🏆 PHASE 2 v2 — BEST MODEL                 │
│                                             │
│  AUC-ROC      :  0.949  🟢 (Target: 0.900) │
│  Accuracy     :  88.6%  🟢                  │
│  Sensitivity  :  91.4%  🟢 (Target: ≥80%)  │
│  Specificity  :  86.3%  🟢                  │
│  Test samples :  299 (clinical, balanced)   │
└─────────────────────────────────────────────┘
```

**Bottom table — full progression:**

| Phase 2 v1 | Phase 2 v2 | Gain |
|-----------|-----------|------|
| AUC: 0.911 | **AUC: 0.949** | **+3.8%** |
| Accuracy: 85.3% | **88.6%** | +3.3% |
| Sensitivity: 80.6% | **91.4%** | **+10.8% 🚀** |

---

## SLIDE 12 — Complete Experiment Journey (All Stages)

**Heading:** 📊 Full Experimental Journey — All 14 Experiments

**Dataset size evolution bar (horizontal):**

```
 Kaggle CXR (5,216)        ████████████████████████
 IU X-Ray (multimodal)     █████
 MIMIC Pilot (139)         █
 MIMIC Scale-Up (1,854)    █████████
 MIMIC Final (1,989)       ██████████
```

**Master results table (key milestones only):**

| Stage | Experiment | Dataset | Architecture | AUC | Key Insight |
|-------|-----------|---------|--------------|-----|-------------|
| Stage 0 | v2 (Kaggle) | 5,216 Kaggle | ResNet50+DSC+GCSA | 92% Acc | Arch validated |
| Stage 1 | Pilot Ph.2 | 139 MIMIC | ResNet50+DSC+GCSA | 0.704 | +3.7% from text |
| Stage 2 | Pilot arch scale | 1,989 MIMIC | ResNet50+DSC+GCSA | **0.713** | ❌ Arch fails |
| Stage 2 | v4 CrossVal | 1,854 MIMIC | DenseNet+CBAM | 0.820±0.021 | Reliable eval |
| Stage 2 | v5 Best img-only | 1,854 MIMIC | DenseNet+CBAM | 0.864 | AUC ceiling |
| Stage 2 | v6 ❌ | 1,989 MIMIC | DenseNet(mimic_nb) | 0.811 | Failed |
| **Stage 3** | **Phase 2 v1** | **1,989 MIMIC+text** | DenseNet+BERT | **0.911** | Text helps |
| **Stage 3** | **Phase 2 v2 🏆** | **1,989 MIMIC+text** | DenseNet+CrossAttn | **0.949** | **Best** |

---

## SLIDE 13 — Lessons Learned & What's Next

**Heading:** What Part 1 Taught Us (6 Hard Lessons)

**Left — 6 lessons:**

1. 🧠 **Wrong pretraining = wrong model** — ResNet50 (ImageNet) failed on 1,989 MIMIC images (AUC 0.713). Must use CXR-domain pretrained backbone.
2. 🏥 **Clinical ≠ Kaggle** — 92% on Kaggle → 0.713 AUC on real MIMIC. Public benchmarks give false confidence.
3. ⚠️ **Change one thing at a time** — v6 changed 4 things at once → impossible to diagnose the failure.
4. 📉 **Small N = unreliable numbers** — N=7 test set (pilot balanced) gave AUC 0.917. On N=293: AUC 0.713. Statistics matter.
5. 🔼 **Image-only has a ceiling** — DenseNet maxed at AUC 0.864 (v5). Can't go further without clinical text.
6. 🔀 **Multimodal fusion genuinely works** — leakage-free text → AUC 0.704→0.911→0.949.

**Right — Part 2 roadmap:**

| Priority | What | Goal |
|----------|------|------|
| 🔥 #1 | External validation (NIH/CheXpert) | Generalisability |
| 🔥 #1 | Expand to 5,000+ MIMIC images | Statistical power |
| ⚡ Soon | Phase 3: Add MIMIC-IV lab values | 3-modality fusion |
| ⚡ Soon | Grad-CAM on all phases | Explainability |
| 🔮 Future | 14-class multi-label | Clinical scope |

---

## SLIDE 14 — Key Artefacts & Code

**Heading:** Reproducibility — All Code, Data & Results on GitHub

**Left — Repository structure:**

```
PneumoFusionNet/
├── model_experiments/          ← Stage 0 (v1, v2, v3-1, v3-2)
│   └── [Kaggle/IU experiments]
├── mimic/
│   ├── mimic_pilot_139/        ← Stage 1 (139-sample pilot)
│   │   ├── Notebooks/ (4 phases)
│   │   └── Scripts/  (3 data scripts)
│   └── 1000_dataset/           ← Stage 2 & 3
│       ├── notebooks_scaleUp/  (10 notebooks)
│       ├── mimic_paired_dataset.csv  (1,989 PA)
│       └── outputs/ (all results + models)
└── docs/
    └── Summary till now (05 july).md
```

**Right — Data access note:**

```
┌──────────────────────────────────────────────────────┐
│  ⚠️ RESTRICTED DATA NOTICE                           │
│                                                      │
│  MIMIC-CXR-JPG (v2.0.0) — PhysioNet                │
│  Access requires:                                    │
│  → CITI Program training (human subjects research)  │
│  → PhysioNet Data Use Agreement (DUA) signed        │
│  → Institutional affiliation verification           │
│                                                      │
│  Public dataset experiments (Stage 0) are            │
│  fully reproducible from GitHub without DUA.         │
│                                                      │
│  🔗 github.com/Ashutosh-yadav0001/PneumoFusionNet   │
└──────────────────────────────────────────────────────┘
```

---

## SLIDE 15 — Conclusion

**Heading:** PneumoFusionNet Part 1 — Honest Findings, Clear Direction

**Large summary block:**

```
  STAGE 0 (Public Kaggle/IU):
  ✅ ResNet50 + DSC + GCSA validated — 92% acc on Kaggle
  ✅ Multimodal pipeline prototyped on IU dataset

  STAGE 1 (MIMIC Pilot, 139 samples):
  ✅ Anti-leakage pipeline built (IMPRESSION stripped)
  ✅ Text fusion: +3.7% AUC leakage-free on imbalanced data
  ⚠️ N=7 test = statistically unreliable (Fisher p≈1.0)

  STAGE 2 (MIMIC Scale-Up, 1,989 PA images):
  ❌ ResNet50 + DSC + GCSA: AUC 0.713 — confirmed pilot arch fails
  ✅ DenseNet-121 (CXR pretrained): AUC 0.820±0.021 (5-fold)
  ✅ Best image-only: AUC 0.864 (v5) — then hit ceiling

  STAGE 3 (MIMIC Multimodal, 1,989 images + reports):
  ✅ Phase 2 v1 (Concat): AUC 0.911 (+6.6% over image-only)
  🏆 Phase 2 v2 (CrossAttn): AUC 0.949, Sensitivity 91.4%
```

**Bottom two-column:**

| Metric | Value |
|--------|-------|
| Final AUC | **0.949** |
| Final Sensitivity | **91.4%** |
| Total experiments | **14** |
| MIMIC images used | **1,989 (PA-only)** |
| Dataset access | CITI + DUA signed |

---

## 📊 CHARTS TO CREATE IN GOOGLE SLIDES

| Chart | Type | Key Data |
|-------|------|----------|
| Dataset size evolution | Horizontal bar | Kaggle 5216 / IU / MIMIC 139 / MIMIC 1854 / MIMIC 1989 |
| AUC progression all stages | Line/bar chart | 0.713→0.812→0.820→0.864→0.911→0.949 |
| Class distribution MIMIC | Bar | 1062 Normal vs 927 Pneumonia (PA, 1989 total) |
| Pilot arch failure | Side-by-side bar | Pilot AUC 0.917 (N=7) vs Scale AUC 0.713 (N=293) |
| Phase 2 improvements | Waterfall/bar | v1 0.911 → v2 0.949 (+3.8%) |
| Sensitivity comparison | Grouped bar | All versions: Sens from 69.7% → 91.4% |

---

## 🎨 DESIGN SYSTEM

| Element | Style |
|---------|-------|
| Background | `#0D1B2A` (dark navy) |
| Headings | Montserrat Bold, white, 32–40pt |
| Body text | Roboto Regular, `#CBD5E0`, 18pt |
| Cyan accent | `#00E5FF` — numbers, borders |
| Warning boxes | `#E67E22` background, white text |
| Success cards | `#2ECC71` background, white text |
| Failure cards | `#E74C3C` background, white text |
| Key numbers | 48–56pt bold white, `#1A5276` card |
