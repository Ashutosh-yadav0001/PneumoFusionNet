# 🎞️ PneumoFusionNet — Part 2 PPT Slide Content
### For Google Slides / PowerPoint | 8–10 min | 15 Slides

> **Design System (same as Part 1):**
> Dark navy background `#0D1B2A`, cyan accent `#00E5FF`,
> orange for warnings `#E67E22`, green for results `#2ECC71`, red for failures `#E74C3C`.
> Font: **Montserrat Bold** headings (32–40pt white), **Roboto** body (18pt `#CBD5E0`).
> Key numbers: 56pt bold white on teal card `#1A5276`.

---

## SLIDE 1 — Title Slide

```
┌──────────────────────────────────────────────────────────────┐
│  [Background: Blurred chest X-ray + glowing fusion diagram]  │
│                                                              │
│  🫁  PneumoFusionNet                                         │
│      Part 2: Image + Text + WBC Fusion                       │
│                                                              │
│      From Cross-Attention to a One-Blood-Test Model          │
│      for Pneumonia Detection                                 │
│                                                              │
│  ──────────────────────────────────────────────────────────  │
│  Ashutosh Yadav  |  Roll: 23035010693                        │
│  B.Sc. (Hons.) Data Science & AI — IIT Guwahati             │
│  Trimester 8 Term Project  |  September 2026                 │
└──────────────────────────────────────────────────────────────┘
```

**Speaker note:** "This is Part 2 of PneumoFusionNet. In Part 1 we built the image-only pipeline and hit a ceiling at AUC 0.826. Today I'll show how adding a radiology report and then just one blood test — the WBC count — pushes the model to AUC 0.971 with 92% sensitivity."

---

## SLIDE 2 — Where Part 1 Left Off

**Heading:** The Part 1 Ceiling — and the Roadmap for Part 2

**Left 50%:**

```
┌─────────────────────────────────────────┐
│  ✅ WHAT PART 1 ACHIEVED                │
│  Architecture: DenseNet-121 + CBAM      │
│  Data: MIMIC-CXR PA view, 1,989 images  │
│  Best AUC:       0.826                  │
│  Sensitivity:    71.0%                  │
│                                         │
│  ❌ WHY IT HIT A CEILING                │
│  X-ray shows lung appearance only.      │
│  Cannot see fever, WBC, patient         │
│  history — information every doctor     │
│  uses to diagnose pneumonia.            │
└─────────────────────────────────────────┘
```

**Right 50%:**

```
  Part 1              Part 2 Journey
  ───────     ──────────────────────────────
  Image  →    Phase 2:  Image + Report Text
  only              AUC: 0.949

              Phase 3c: Image + Text + WBC  ⭐
                    AUC: 0.971

              Next:     Indian hospital data
```

**Speaker note:** "Part 1 showed that even the best image model gets stuck. The X-ray cannot tell us whether the immune system is fighting an infection. Part 2 adds two things already available in every hospital: the leakage-free radiology report, and a routine blood test WBC count."

---

## SLIDE 3 — The Anti-Leakage Rule (Key Concept)

**Heading:** The Rule We Never Break — No IMPRESSION

**Centre large card:**

```
┌──────────────────────────────────────────────────────────────┐
│                  A Real Radiology Report                     │
│                                                              │
│  FINDINGS:   Increased opacity in the right lower lobe.      │
│              Mild blunting of the costophrenic angle.        │
│                                                              │
│  HISTORY:    65-year-old male, fever 3 days, cough.          │
│                                                              │
│  IMPRESSION: PNEUMONIA — right lower lobe consolidation.     │
│                                                              │
│  ✅  We use:   FINDINGS + HISTORY                            │
│  ❌  We strip: IMPRESSION  ← this IS the diagnosis           │
└──────────────────────────────────────────────────────────────┘
```

**Bottom (orange):** "90%+ of multimodal papers accidentally read IMPRESSION — the model memorises the label, not the diagnosis."

**Speaker note:** "The most critical design decision: every report ends with IMPRESSION — a one-line diagnosis by the radiologist. If the model sees that, it reads the answer. We strip it from every single report. Our model never sees IMPRESSION."

---

## SLIDE 4 — Phase 2: Adding the Radiology Report

**Heading:** Phase 2 — Image + Leakage-Free Report Text

**Left 45%:**

```
  Phase 2v1 (concat baseline)
  ┌──────────────────────────────┐
  │  Image 1024 + BERT [CLS]    │
  │  CONCAT → MLP → diagnosis   │
  │  AUC: 0.911  Sens: 80.6%   │
  └──────────────────────────────┘
        ⬇ sensitivity too low

  Phase 2v2 (7 improvements)
  ┌──────────────────────────────┐
  │  Image queries BERT tokens   │
  │  8-head Cross-Attention      │
  │  + Focal Loss + Mixup        │
  │  AUC: 0.949  Sens: 91.4%  ✅│
  └──────────────────────────────┘
```

**Right 55%:**

```
  7 Changes from v1 → v2:
  ① FINDINGS only → FINDINGS + HISTORY
  ② BERT frozen → last 2 layers unfrozen
  ③ Concat → 8-head CrossAttention
  ④ CrossEntropy → Focal Loss (γ=2)
  ⑤ Standard → Mixup on embeddings (α=0.2)
  ⑥ Single LR → separate (BERT=1e-5, head=2e-4)
  ⑦ Added Grad-CAM visualisations
```

**Speaker note:** "Phase 2 has two versions. V1 simply concatenated text and image — it helped, but mostly improved Normal detection. V2 replaced concatenation with cross-attention: the image features look at every word of the report and decide which ones are relevant. That is where the sensitivity jump came from — 80 to 91 percent."

---

## SLIDE 5 — Cross-Attention: How It Works

**Heading:** Image Queries the Report — Word by Word

**Centre:**

```
  Image features (1024-d)        Report tokens (768-d × N words)
        │                                    │
        ▼  Q (Query)           K, V (Key, Value) ▼
  ┌──────────────────────────────────────────────────┐
  │  Score = Q · Kᵀ / √64   →   softmax weights      │
  │  Output = weighted sum of Value vectors           │
  │                                                   │
  │  "Which words in the report match                 │
  │   what the image is showing?"                    │
  └──────────────────────────────────────────────────┘
                      │
               512-d context vector
         (image-anchored text summary)
```

**Bottom 3 cards:**

```
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │  8 attention   │  │  Image is the  │  │  All 512 token │
  │  heads running │  │  QUERY — it    │  │  positions     │
  │  in parallel   │  │  searches the  │  │  used, not     │
  │                │  │  report        │  │  just [CLS]    │
  └────────────────┘  └────────────────┘  └────────────────┘
```

**Speaker note:** "The image feature vector becomes a 'query' — it looks at every word in the report and assigns attention weights. Words like 'consolidation' and 'opacity' get high weight. This produces a 512-dimensional summary that is specifically relevant to what the image shows — not a generic text summary."

---

## SLIDE 6 — Phase 2 Results

**Heading:** Phase 2 Results — Radiology Text Adds +12% AUC

**Table:**

```
┌─────────────────────────────────────────────────────────┐
│  Model              AUC    Sens.   Spec.   Acc.          │
│  ─────────────────────────────────────────────────────  │
│  Phase 1 (image)   0.826   71.0%    —     76.3%          │
│  Phase 2v1         0.911   80.6%   89.4%  85.3%          │
│  Phase 2v2 ✅      0.949   91.4%   86.3%  88.6%          │
│  Scale-up (3,763)  0.946   90.3%   89.1%  87.8%          │
└─────────────────────────────────────────────────────────┘
```

**3 highlight cards (green):**

```
  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
  │  +12.3%        │  │  91.4%         │  │  Stable on     │
  │  AUC gain      │  │  Sensitivity   │  │  3,763 imgs    │
  │  vs image-only │  │  9 in 10       │  │  AUC drops     │
  │                │  │  caught        │  │  only 0.003    │
  └────────────────┘  └────────────────┘  └────────────────┘
```

**Speaker note:** "AUC jumps from 0.826 to 0.949 — a 12 point gain. More importantly, sensitivity goes from 71 to 91 percent. We also tested on a doubled dataset of 3,763 images and the AUC barely changes — 0.946 versus 0.949. The model generalises, it is not overfitting."

---

## SLIDE 7 — Phase 3c: Adding WBC

**Heading:** Phase 3c — X-Ray + Report + One Blood Test

**Left 55% — architecture sketch:**

```
  Image          Report (no IMPRESSION)   WBC count
    │                    │                    │
 DenseNet+CBAM    ClinicalBERT          MLP 1→128→64
    │                    │                    │
 1024-d           768-d tokens             64-d
    │                    │
    └───── CrossAttn ────┘
              512-d
                │
    concat [1024 + 512 + 64 = 1600-d]
                │
     MLP → Normal / Pneumonia
```

**Right 45%:**

```
  ┌──────────────────────────────────────┐
  │  🩸 Why WBC?                        │
  │                                      │
  │  Routine CBC blood test              │
  │  Ready in 15 min of ED arrival       │
  │                                      │
  │  < 4,000   →  immune suppressed      │
  │  4–11,000  →  normal                 │
  │  > 11,000  →  infection likely  ✅   │
  │  > 20,000  →  severe infection       │
  │                                      │
  │  We pass it through a 3-layer MLP    │
  │  to learn non-linear thresholds      │
  └──────────────────────────────────────┘
```

**Speaker note:** "We add one extra input: the WBC count. The image shows what the lung looks like. WBC tells us what the body is doing about it. We pass WBC through a small network rather than raw because WBC 12 and WBC 25 are clinically very different — not just 'more'."

---

## SLIDE 8 — Phase 3c Results

**Heading:** Phase 3c — AUC 0.971, Sensitivity 92.3%

**Large result card:**

```
┌──────────────────────────────────────────────────────────┐
│   Phase 3c:  X-ray  +  Report Text  +  WBC              │
│   Test: 565 patients | MIMIC-CXR 3,763-image scale-up   │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │  AUC     │  │  Sens.   │  │  Spec.   │  │  Acc.  │  │
│  │  0.971   │  │  92.3%   │  │  93.6%   │  │  92.9% │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│                                                          │
│  +2.5% AUC over text-only  |  +5.7% Sensitivity         │
└──────────────────────────────────────────────────────────┘
```

**Cut-off table:**

```
  Cut-off Setting          Acc.    Sens.    Spec.
  Default (0.50)           92.4%   94.7%    90.0%  ← best for screening
  Best-balance (0.559)     92.9%   92.3%    93.6%  ← best overall ✅
  Screening (0.575)        92.7%   91.2%    94.3%  ← fewest false alarms
```

**Speaker note:** "AUC 0.971. The model catches 92.3 percent of pneumonia patients and correctly clears 93.6 percent of normal patients. We can tune the cut-off: the default gives 94.7 percent sensitivity for a screening-first setting where missing sick patients is the biggest risk."

---

## SLIDE 9 — The Ablation Table

**Heading:** Ablation — Every Input Earns Its Place

**Table:**

```
┌────────────────────────────────────────────────────────────┐
│  Phase     What goes in          AUC    Sens.  Spec.  Acc. │
│  ─────────────────────────────────────────────────────     │
│  Phase 1   X-ray only            0.826  71.0%   —    76.3% │
│  Phase 2   X-ray + Report        0.946  86.6%  89.1% 87.8% │
│  Phase 3c  X-ray + Report + WBC  0.971  92.3%  93.6% 92.9% │
│  ─────────────────────────────────────────────────────     │
│  Adding text:    +12.0% AUC    +15.6% Sensitivity          │
│  Adding WBC:      +2.5% AUC    + 5.7% Sensitivity          │
└────────────────────────────────────────────────────────────┘
```

**Progress bar:**

```
  AUC  0.82 ──────────────────────────────────────── 1.00
       0.826  ──────────────►  0.946  ────►  0.971
              +12.0% (text)        +2.5% (WBC)
```

**Speaker note:** "Each row adds exactly one thing. The biggest jump is from the report text — 12 percentage points. Adding WBC gives another 2.5 points. Together: 0.826 to 0.971, a 14.5 point total gain. Every input earns its place."

---

## SLIDE 10 — WBC vs Full 17-Feature Model

**Heading:** Do We Need 17 Lab Tests? No — Just WBC.

**Table:**

```
┌──────────────────────────────────────────────────────────┐
│  Model         Extra data          AUC    Sens.   Spec.  │
│  ───────────────────────────────────────────────────     │
│  Phase 2v2     None (text only)    0.946  86.6%   89.1%  │
│  Phase 3c ✅   WBC (1 test)        0.971  92.3%   93.6%  │
│  Phase 3 full  17 lab features     0.989  89.1%   94.3%  │
└──────────────────────────────────────────────────────────┘
```

**Two boxes:**

```
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ 🏥 Full 17-feature model    │  │ ✅ Phase 3c — WBC only      │
│ Creatinine, CRP, albumin,   │  │ One CBC blood test           │
│ vitals, alk.phos...         │  │                             │
│ Ready: 1–4 hours            │  │ Ready: 15 minutes           │
│ AUC: 0.989                  │  │ AUC: 0.971                  │
│ Sensitivity: 89.1%          │  │ Sensitivity: 92.3%  ✅      │
│ ← misses more sick patients │  │ ← catches MORE sick patients│
└─────────────────────────────┘  └─────────────────────────────┘
```

**Caption (cyan):** "Phase 3c closes 92.5% of the AUC gap using 1 of 17 features."

**Speaker note:** "This is the most surprising finding. The full 17-feature model needs creatinine, CRP, albumin — tests that take hours. Our WBC-only model needs one result ready in 15 minutes. The AUC gap is only 0.018. But Phase 3c actually has HIGHER sensitivity — 92.3 versus 89.1 percent. It catches more sick patients."

---

## SLIDE 11 — Why This Works Clinically

**Heading:** The Clinical Logic Behind the WBC Gain

**Diagram:**

```
  X-ray shows:                     WBC shows:
  ┌──────────────────────┐         ┌──────────────────────┐
  │ "Opacity in the      │         │ "WBC = 18,000/µL     │
  │  right lower lobe"   │  +      │  Body is fighting    │
  │  (ambiguous)         │         │  an infection"       │
  └──────────────────────┘         └──────────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │  HIGH confidence: PNEUMONIA  │
              │  (not fluid or collapse)     │
              └──────────────────────────────┘
```

**Bottom box (orange):** "A shadow on X-ray alone is ambiguous. A shadow + WBC 18,000 is almost certainly pneumonia. Our model learns this logic from data — we do not hard-code it."

**Speaker note:** "The X-ray tells us what the lung looks like. WBC tells us whether the body is mounting an immune response. A shadow could be pneumonia, atelectasis, or pulmonary oedema. But a shadow plus an elevated WBC strongly points to infection. This is exactly how a doctor thinks — and now our model thinks the same way."

---

## SLIDE 12 — Limitations

**Heading:** Honest Limitations

**4 cards:**

```
┌──────────────────────┐  ┌──────────────────────┐
│ 🏥 Single Hospital   │  │ 🔢 Binary Task Only  │
│ All data from BIDMC  │  │ Normal vs Pneumonia  │
│ (US). Results may    │  │ only. Real X-rays    │
│ not generalise to    │  │ involve 14+ findings.│
│ Indian hospitals.    │  │                      │
└──────────────────────┘  └──────────────────────┘

┌──────────────────────┐  ┌──────────────────────┐
│ 🧪 WBC Imputation    │  │ ✅ Data Ethics       │
│ ~15% of WBC values   │  │ All MIMIC-CXR use    │
│ filled with training │  │ complied with        │
│ set median.          │  │ PhysioNet DUA and    │
│                      │  │ CITI ethics training.│
└──────────────────────┘  └──────────────────────┘
```

**Speaker note:** "We cannot claim this works everywhere. All data is from one American hospital. TB co-infection is common in India and changes how pneumonia looks on an X-ray. Malnutrition affects WBC patterns. We do not yet know how the model performs there — that is exactly what the next phase will test."

---

## SLIDE 13 — Next Phase: Indian Hospital Data

**Heading:** Next Phase — Validate on Indian Clinical Data

**Roadmap:**

```
  DONE ✅                                   NEXT 🔜
  ──────────────────────────────────────────────────────────
  Phase 1:  X-ray only (MIMIC-CXR, USA)
            AUC: 0.826

  Phase 2:  X-ray + Report (MIMIC-CXR)
            AUC: 0.949

  Phase 3c: X-ray + Report + WBC (MIMIC-CXR) ← We are here
            AUC: 0.971

  Phase 4:  Collect Indian hospital dataset  ← NEXT
            PA chest X-rays + WBC from CBC
            Fine-tune Phase 3c
            Test generalisation
```

**Right box:**

```
  ┌────────────────────────────────────────┐
  │  Why India matters:                    │
  │                                        │
  │  • TB co-infection → different X-ray   │
  │    patterns from US pneumonia          │
  │  • Malnutrition → altered WBC ranges   │
  │  • Scanner variety → different image   │
  │    quality from MIMIC-CXR standard     │
  │  • Underserved setting → real impact   │
  └────────────────────────────────────────┘
```

**Speaker note:** "For the next phase, we want to collect chest X-rays paired with WBC counts from Indian hospitals. India has a very different disease profile. If the model works there, it has real clinical value for resource-limited settings where waiting hours for a full lab panel is not always possible."

---

## SLIDE 14 — Full Project Journey

**Heading:** The Complete PneumoFusionNet Journey

**Timeline:**

```
  PART 1                                  PART 2
  ────────────────────────────────────────────────────────────
  Stage 0     Stage 1    Stage 2     Phase 2    Phase 3c
  Kaggle/IU   MIMIC      MIMIC       X-ray +    X-ray +
  Arch test   Pilot      Scale-up    Report     Report + WBC
  (valida-    (139)      (1,989)     CrossAttn
  tion)

  AUC 0.97   AUC 0.85  AUC 0.826  AUC 0.949  AUC 0.971 ✅
  (Kaggle)   (MIMIC)   (MIMIC)    (MIMIC)    (MIMIC 3,763)
```

**3 lessons:**

```
  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │ Lesson 1:          │  │ Lesson 2:           │  │ Lesson 3:          │
  │ Domain matters.    │  │ Anti-leakage is     │  │ One good biomarker │
  │ ImageNet weights   │  │ non-negotiable.     │  │ beats 17 weak ones │
  │ fail on clinical   │  │ Strip IMPRESSION    │  │ when time is the   │
  │ X-rays.            │  │ every time.         │  │ constraint.        │
  └────────────────────┘  └────────────────────┘  └────────────────────┘
```

**Speaker note:** "Three core lessons from the whole project. First, domain pretraining matters enormously — ResNet on ImageNet fails, DenseNet on real chest X-rays works. Second, the anti-leakage rule is non-negotiable. Third, one well-chosen biomarker is more useful than many poorly-available lab values."

---

## SLIDE 15 — Conclusion & Thank You

**Heading:** Summary

**Results table:**

```
┌──────────────────────────────────────────────────────────────┐
│  Input                AUC     Sensitivity   Specificity      │
│  ────────────────────────────────────────────────────────   │
│  X-ray only          0.826      71.0%            —           │
│  + Report text       0.949      91.4%          86.3%         │
│  + WBC (1 test) ✅   0.971      92.3%          93.6%         │
│  Total gain:        +14.5%     +21.3%                        │
└──────────────────────────────────────────────────────────────┘
```

**5 takeaways:**

```
  ① X-ray alone hits a ceiling — clinical context breaks it.
  ② Leakage-free report text gives the biggest single AUC jump.
  ③ WBC adds +5.7% sensitivity with 1 blood test in 15 minutes.
  ④ Phase 3c beats the 17-feature model on sensitivity.
  ⑤ Next: validate on Indian hospital data.
```

**Footer:**

```
  Ashutosh Yadav  |  Roll: 23035010693  |  IIT Guwahati
  github.com/Ashutosh-yadav0001/PneumoFusionNet
  MIMIC-CXR: restricted (PhysioNet DUA + CITI required)
```

**Speaker note:** "To summarise: Part 2 shows you can go from AUC 0.826 to 0.971 by adding two things already in every hospital — the leakage-free report and a WBC count. The model is practical. No special equipment needed. The next step is Indian clinical data. Thank you."
