# PneumoFusionNet — Video Presentation Script
### IIT Guwahati | Term Project (Trimester 8) | Ashutosh Yadav (23035010693)
### Estimated Duration: ~8–10 minutes (45–60 sec per slide)

---

## 🎬 SLIDE 1 — Title (45 sec)

*[Show title slide. Speak clearly and confidently.]*

> "Hello, my name is Ashutosh Yadav, Roll Number 23035010693, from the
> B.Sc. Data Science and Artificial Intelligence programme at IIT Guwahati.
>
> This is my Term 8 project presentation on **PneumoFusionNet** — a multimodal
> deep learning framework for pneumonia detection using chest X-rays and
> clinical reports.
>
> This work is inspired by the PneumoFusion-Net paper published in
> Frontiers in Physiology in March 2025 by Yu et al.
>
> Let me walk you through the complete research journey — from validating
> our architecture on public datasets, to accessing restricted clinical data,
> to the surprising failure that changed our direction."

---

## 🎬 SLIDE 2 — Problem & Motivation (60 sec)

*[Point to the three boxes — challenge, gap, solution.]*

> "Pneumonia kills approximately 2.5 million people every year.
> Chest X-ray is the primary diagnostic tool because it's fast and affordable.
>
> But reading a chest X-ray is genuinely difficult. The same cloudy area can
> look like pneumonia, fluid buildup, or a collapsed lung section. Even expert
> radiologists disagree about 20% of the time.
>
> Existing AI models are **image-only** — they ignore the clinical context that
> every real doctor uses: patient symptoms, lab results, and the written
> radiology report.
>
> Adding the report text should help resolve this ambiguity. But there's a
> critical trap called **label leakage** — if the model reads the radiologist's
> final conclusion during training, it simply memorises the answer instead
> of actually learning to diagnose.
>
> Our solution — PneumoFusionNet — handles this with a strict anti-leakage
> pipeline and uses the restricted MIMIC-CXR hospital database."

---

## 🎬 SLIDE 3 — Architecture: ResNet50 + DSC + GCSA (60 sec)

*[Point to each of the three component boxes from left to right.]*

> "Before touching any clinical data, we designed and validated a custom
> image backbone consisting of three components.
>
> **First, ResNet50** — a standard deep learning backbone pretrained on
> ImageNet. We adapted its first convolutional layer to accept single-channel
> grayscale X-ray images by averaging the pretrained RGB weights.
>
> **Second, Depthwise Separable Convolution — DSC** — this compresses the
> 2048-channel ResNet output to 1024 channels using approximately 50% fewer
> parameters compared to a standard convolution. The mathematical formulation
> is shown on screen.
>
> **Third, Global Context Spatial Attention — GCSA** — this tells the network
> *what* features to focus on through channel attention, and *where* in the
> image to look through spatial attention. Together, they help the model
> concentrate on the actual pathological regions in the lung.
>
> The final output is a **1024-dimensional image feature vector**."

---

## 🎬 SLIDE 4 — Stage 0 Results + Dataset Search (60 sec)

*[Point to results table on left, then dataset comparison on right.]*

> "We validated this architecture on public datasets first.
>
> On the **Kaggle Chest X-Ray dataset** — 5,216 images — we achieved 92%
> accuracy and F1 of 0.91. The IU X-Ray experiments confirmed that our
> multimodal pipeline code ran correctly end-to-end.
>
> However — and this is important — 92% on Kaggle only confirms the code
> is correct. It says nothing about clinical readiness. Kaggle data is
> pre-cleaned, balanced, and far too easy compared to real hospital data.
>
> So we searched for a better dataset. We needed three things simultaneously:
> chest X-ray images, clinical history, and full radiology reports with
> separate FINDINGS and IMPRESSION sections.
>
> After evaluating Kaggle, IU, NIH, and CheXpert, we identified **MIMIC-CXR-JPG**
> as the only dataset meeting all requirements.
>
> But MIMIC-CXR is **not publicly downloadable**. Accessing it required
> completing CITI Program ethics training and signing a PhysioNet
> Data Use Agreement."

---

## 🎬 SLIDE 5 — Anti-Leakage Pipeline (60 sec)

*[Point to the leakage explanation first, then the pipeline diagram.]*

> "Before running a single experiment on MIMIC-CXR, we built our
> anti-leakage data pipeline. This is the most critical design decision
> in the entire project.
>
> Every radiology report has an IMPRESSION section — the radiologist's
> final written diagnosis. If our model ever reads this during training,
> it's not learning to diagnose — it's just reading the answer.
>
> We developed a script called **rebuild_dataset_findings.py** that
> parses every report, retains the INDICATION and FINDINGS sections
> which contain pre-diagnosis clinical evidence, and completely strips
> the IMPRESSION and CONCLUSION sections.
>
> The system architecture works in two phases shown on the right:
> The image branch passes through ResNet50, DSC, and GCSA to produce a
> 1024-dimensional vector.
> The text branch feeds the leakage-free report into Bio_ClinicalBERT —
> a BERT model pretrained on millions of clinical notes — producing a
> 768-dimensional vector.
>
> Both vectors are concatenated into 1792 dimensions and classified
> by a fusion MLP."

---

## 🎬 SLIDE 6 — Stage 1: Pilot Results (60 sec)

*[Walk through the table row by row, highlight the gain row in green.]*

> "For the pilot study, we extracted 139 samples from MIMIC-CXR —
> 118 Normal and 21 Pneumonia — giving a 5.6 to 1 class imbalance.
> We ran four systematic phases.
>
> Phase 1 — image only, realistic imbalanced split: AUC of 0.667.
>
> Phase 1.1 — image only, artificially balanced: AUC jumps to 0.917.
> But as we'll see, this number is misleading.
>
> Phase 2 — image plus leakage-free text, imbalanced: AUC improves
> to **0.704** — that's a genuine **+3.7% gain** over Phase 1.
>
> Phase 2.1 — image plus text, balanced: AUC of 0.917 again.
>
> The key result is in the green row — on the **realistic imbalanced split**,
> adding text from FINDINGS and INDICATION — with IMPRESSION completely
> stripped — genuinely helped the model.
>
> This proves that the clinical text contains real diagnostic signal
> beyond what the image shows."

---

## 🎬 SLIDE 7 — The N=7 Problem (50 sec)

*[Speak with emphasis on the statistical point.]*

> "But here is where scientific honesty becomes critical.
>
> The balanced test set had exactly **seven samples**. One misclassification
> changes the accuracy by 14.28 percent.
>
> Using the standard 95% confidence interval formula, Phase 1.1's accuracy
> of 85.7% has a confidence interval of **59.8% to 100%** — an enormous range.
>
> A Fisher's Exact Test gives p approximately equal to 1.0 — the difference
> between phases is completely statistically insignificant.
>
> The AUC of 0.917 on 7 samples is **statistically meaningless**.
>
> Documenting this gap honestly is one of the most important scientific
> contributions of Part 1. We cannot make clinical claims from 7 test samples.
> We must scale up — and that is exactly what we did next."

---

## 🎬 SLIDE 8 — Scale-Up Failure (60 sec)

*[Pause briefly on the failure numbers to let them land.]*

> "For the scale-up, we curated **1,989 PA-only images** from MIMIC-CXR —
> 1,062 Normal and 927 Pneumonia.
>
> Two critical improvements over the pilot:
> First, we restricted to **PA-only views** — posteroanterior frontal X-rays —
> eliminating the view-type confound where the model could learn
> 'AP view equals sicker patient' instead of actual lung pathology.
>
> Second, we used **patient-level splits** — GroupShuffleSplit by patient ID —
> guaranteeing zero patient overlap between train, validation, and test sets.
>
> We then ran the **identical** ResNet50 + DSC + GCSA architecture.
>
> The results on the right show the honest answer:
> AUC dropped from 0.917 to **0.713**.
> Accuracy dropped from 85.7% to **66.2%**.
> The model misses 30% of pneumonia patients — sensitivity of only 69.9%.
>
> The root cause is clear: ImageNet pretraining is fundamentally mismatched
> for clinical-scale chest X-ray pathology detection."

---

## 🎬 SLIDE 9 — Key Findings (45 sec)

*[Point to each of the four numbered boxes.]*

> "Part 1 delivered four critical findings.
>
> **Finding 1:** Public benchmarks do not predict clinical performance.
> 92% on Kaggle became 66% on real hospital data.
>
> **Finding 2:** The anti-leakage text pipeline works.
> Stripping IMPRESSION and using only FINDINGS produced a genuine +3.7% AUC gain.
>
> **Finding 3:** Small test sets are statistically meaningless.
> AUC of 0.917 on 7 samples has a confidence interval spanning the entire
> range from 59% to 100%.
>
> **Finding 4:** The architecture must change.
> ResNet50 with ImageNet pretraining cannot handle clinical-scale CXR
> pathology detection. A different backbone is required."

---

## 🎬 SLIDE 10 — Conclusion & Part 2 Roadmap (60 sec)

*[Speak with forward-looking energy. End strongly.]*

> "To summarise Part 1 — we delivered three contributions:
> a robust anti-leakage data pipeline, rigorous proof that public dataset
> results don't transfer to clinical environments, and an honest failure
> at scale that tells us exactly what to do next.
>
> The solution is on the right — **DenseNet-121** from the torchxrayvision
> library, pretrained on over 500,000 actual chest X-rays from CheXpert,
> NIH ChestX-ray14, and MIMIC-CXR itself. Every weight in this network was
> learned from real radiological images. This directly solves the domain
> mismatch failure we observed.
>
> For Part 2, we will scale to the full 1,989-image dataset with patient-level
> splits, add Bio_ClinicalBERT text fusion with cross-attention, incorporate
> MIMIC-IV lab values as a third modality, and use 5-fold GroupKFold
> cross-validation for statistically robust evaluation.
>
> Thank you for watching. The full code is available on GitHub, and all
> MIMIC-CXR data access follows the PhysioNet DUA requirements.
>
> I'm happy to take any questions."

---

## 📋 Presentation Summary

| Slide | Topic | Duration |
|-------|-------|----------|
| 1 | Title | 45 sec |
| 2 | Problem & Motivation | 60 sec |
| 3 | Architecture: ResNet50 + DSC + GCSA | 60 sec |
| 4 | Stage 0 Results + Dataset Search | 60 sec |
| 5 | Anti-Leakage Pipeline | 60 sec |
| 6 | Pilot Results (139 samples) | 60 sec |
| 7 | Statistical Honesty: N=7 Problem | 50 sec |
| 8 | Scale-Up Failure (AUC 0.713) | 60 sec |
| 9 | Key Findings | 45 sec |
| 10 | Conclusion & Part 2 Roadmap | 60 sec |
| **Total** | | **~9 min** |

---

## 🎙️ Recording Tips
- **Pace:** Speak at 120–130 words/minute. Do not rush.
- **Pauses:** Pause 1–2 seconds after each key number (0.713, +3.7%, N=7).
- **Emphasis:** Bold words in script = stress these in speech.
- **Tone:** Confident when presenting results; measured when explaining the failure.
- **Screen:** Advance slide 0.5 seconds *before* you start narrating it.
