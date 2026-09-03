# PneumoFusionNet — Part 2 Video Presentation Script
### IIT Guwahati | Term Project (Trimester 8) | Ashutosh Yadav (23035010693)
### Estimated Duration: ~8–10 minutes (~40–60 sec per slide)

---

## 🎬 SLIDE 1 — Title (45 sec)

*[Show title slide. Speak clearly and confidently. Slight pause after your name.]*

> "Hello, my name is Ashutosh Yadav, Roll Number 23035010693, from the
> B.Sc. Data Science and Artificial Intelligence programme at IIT Guwahati.
>
> This is Part 2 of my Term 8 project on **PneumoFusionNet** — a system for
> detecting pneumonia by combining chest X-rays, hospital report text, and
> a simple blood test result.
>
> In Part 1 we proved that an image-only model hits a hard ceiling at AUC 0.826.
> Today I will show you why that ceiling exists, what we added to break it,
> and how we reached AUC 0.971 using inputs that are already available
> in any hospital within 30 minutes of a patient arriving."

---

## 🎬 SLIDE 2 — Where Part 1 Left Off (50 sec)

*[Point to the Part 1 summary box on the left, then sweep to the roadmap on the right.]*

> "Let me quickly recap where Part 1 ended.
>
> We built a DenseNet-121 model with CBAM attention on 1,989 MIMIC-CXR
> chest X-rays. The best we could achieve was AUC 0.826 and sensitivity 71 percent.
>
> The reason is simple: **the X-ray only shows what the lung looks like.**
> It cannot show whether the patient has a fever, how long they have been sick,
> or whether their white blood cell count is elevated. These are the things a
> real doctor uses to make the final call.
>
> Part 2 adds two pieces of information that fill this gap —
> the written radiology report and a WBC blood count —
> one at a time, so we can see exactly how much each one helps."

---

## 🎬 SLIDE 3 — The Anti-Leakage Rule (55 sec)

*[Point to the three sections of the report box. Emphasise IMPRESSION in red.]*

> "Before I talk about the model, I need to explain the most important design
> decision in this entire project — the anti-leakage rule.
>
> Every radiology report has three sections.
> **FINDINGS** — what the radiologist sees on the image.
> **HISTORY** — why the patient came in.
> **IMPRESSION** — a single line stating the diagnosis.
>
> That IMPRESSION line is the label. If we let the model read it during training,
> it is not learning to diagnose — it is just memorising the answer. This is
> called label leakage, and over 90 percent of multimodal papers in the
> literature fall into this trap.
>
> We strip IMPRESSION from every single report before it touches the model.
> Our model uses only FINDINGS and HISTORY — the same information a doctor
> would use *before* writing the diagnosis."

---

## 🎬 SLIDE 4 — Phase 2: Adding the Report (60 sec)

*[Point to Phase 2v1 box first, then draw attention to the arrow down to 2v2.]*

> "Phase 2 has two versions, and the difference between them is important.
>
> Version 1 was straightforward: take the image features, take a single BERT
> summary number called the CLS token, stick them together, and classify.
> This gave AUC 0.911. But sensitivity was only 80.6 percent —
> it mostly just got better at identifying Normal cases, not at catching
> actual pneumonia patients.
>
> So we made seven improvements for Version 2.
> The most important one is replacing simple concatenation with
> **8-head Cross-Attention** — instead of using one summary number from BERT,
> we let the image look at every single word in the report.
>
> We also fine-tuned the last two BERT layers, switched to Focal Loss
> to focus on the hard cases, and used Mixup data augmentation on the
> embedding space. The result: AUC 0.949, sensitivity 91.4 percent.
> That is 9 in 10 pneumonia patients correctly identified."

---

## 🎬 SLIDE 5 — Cross-Attention (50 sec)

*[Point to the equation area, then to the three cards at the bottom.]*

> "Let me quickly explain how cross-attention works, because it is the
> core of Phase 2 and also of Phase 3c.
>
> The image features become a **Query** — think of it as a question.
> The report words become **Keys and Values** — think of them as possible answers.
>
> The model computes an attention score between the image query and each
> word in the report. Words like 'consolidation', 'opacity', and 'fever'
> get high scores. Common words like 'the' or 'patient' get low scores.
>
> The output is a 512-dimensional weighted summary of the report that is
> specifically tailored to what the image is showing — not a generic
> average of all the words.
>
> This is fundamentally different from the v1 approach of just using
> the CLS token, and it is why sensitivity improved so dramatically."

---

## 🎬 SLIDE 6 — Phase 2 Results (40 sec)

*[Point to the table row by row, ending on the scale-up row.]*

> "Here are the Phase 2 numbers. You can see the progression clearly.
>
> Image only: AUC 0.826.
> Phase 2 Version 1: AUC 0.911. A gain, but sensitivity still low.
> Phase 2 Version 2: AUC 0.949, sensitivity 91.4 percent.
>
> We then doubled the dataset to 3,763 images and tested again.
> AUC barely dropped — from 0.949 to 0.946. The model is not overfitting.
> It genuinely learned the relationship between images and leakage-free text."

---

## 🎬 SLIDE 7 — Phase 3c: Adding WBC (60 sec)

*[Trace each branch of the architecture from top to bottom, then point to the WBC explanation box.]*

> "Phase 3c is the main contribution of Part 2.
>
> We take the Phase 2 model — which is already strong at 0.946 AUC —
> and add one more input: the **WBC count** from a routine Complete Blood Count.
> This is a standard blood test. In any hospital emergency department,
> the result is ready within 15 minutes of the patient arriving.
>
> Look at the architecture. Three branches feed into the model.
> The image goes through DenseNet and produces 1024 numbers.
> The report goes through ClinicalBERT with cross-attention and produces 512.
> The WBC count goes through a small 3-layer network and produces 64 numbers.
> All three are joined into a 1600-dimensional vector and classified.
>
> Why not just pass the raw WBC number directly? Because WBC of 12 thousand
> and WBC of 25 thousand are clinically very different — borderline elevated
> versus severely elevated. A small network can learn these thresholds
> from the data. A raw number cannot."

---

## 🎬 SLIDE 8 — Phase 3c Results (50 sec)

*[Point to the four result numbers one by one, then to the cut-off table.]*

> "Here are the Phase 3c results.
>
> AUC: 0.971. Sensitivity: 92.3 percent. Specificity: 93.6 percent.
> Accuracy: 92.9 percent. Tested on 565 patients from the MIMIC-CXR scale-up set.
>
> Adding just one WBC value to the Phase 2 model gave us plus 2.5 percent AUC,
> plus 5.7 percent sensitivity, and plus 4.5 percent specificity.
>
> We also tested three different cut-off strategies. The default threshold
> gives 94.7 percent sensitivity — safest for screening, where missing
> a sick patient is the biggest risk. The best-balance threshold gives
> 92.9 percent accuracy with a good balance of both. The screening
> threshold pushes specificity to 94.3 percent if reducing false alarms
> is the priority."

---

## 🎬 SLIDE 9 — Ablation Table (45 sec)

*[Trace down the table row by row, then point to the gain numbers at the bottom.]*

> "This table tells the whole story of Part 2 in four lines.
>
> Phase 1, image only: AUC 0.826, sensitivity 71 percent.
> Phase 2, adding the report: AUC 0.946, sensitivity 86.6 percent.
> Phase 3c, adding WBC: AUC 0.971, sensitivity 92.3 percent.
>
> The gain from text: plus 12 percentage points of AUC.
> The gain from WBC: plus 2.5 more, from just one blood test.
>
> Total journey: 0.826 to 0.971 — a 14.5 point improvement.
> Every single input we added earns its place."

---

## 🎬 SLIDE 10 — WBC vs 17-Feature Model (55 sec)

*[Point to Phase 3c vs Phase 3 full, then draw attention to the sensitivity comparison.]*

> "This is the most surprising result in the whole project.
>
> We also trained a Phase 3 model using all 17 available lab and vital sign
> features — creatinine, CRP, albumin, respiratory rate, the full panel.
> That model scores AUC 0.989.
>
> Our Phase 3c model, with just WBC, scores AUC 0.971.
> The gap is only 0.018 — less than 2 percent.
>
> But look at sensitivity. Phase 3c scores 92.3 percent.
> The full 17-feature model scores 89.1 percent.
> **Phase 3c catches more sick patients than the full model.**
>
> And here is why this matters practically: those 17 lab values require
> 1 to 4 hours to collect. WBC from a routine CBC is ready in 15 minutes.
> Phase 3c closes 92.5 percent of the performance gap using only 1 of
> 17 features, while being ready 3 hours faster."

---

## 🎬 SLIDE 11 — Why This Works Clinically (45 sec)

*[Point to the two input boxes, then to the combined output box.]*

> "Let me explain the clinical intuition behind this result.
>
> A shadow on a chest X-ray is genuinely ambiguous. It could be pneumonia,
> a collapsed section of lung, or fluid from heart failure.
> The radiology report helps — it tells us what the radiologist saw and
> the patient's history. But there is still uncertainty.
>
> WBC resolves that uncertainty. If the WBC is above 11,000 per microlitre,
> the body is actively fighting an infection. Combined with a shadow on
> the X-ray, that is almost certainly pneumonia — not fluid or collapse.
>
> Our model learns this reasoning from the data. We did not hard-code it.
> The WBC embedding network and the cross-attention weights together
> learned the clinical logic that doctors use."

---

## 🎬 SLIDE 12 — Limitations (40 sec)

*[Point to each of the four cards.]*

> "I want to be honest about what we cannot claim from this work.
>
> First, all our data comes from one American hospital — Beth Israel
> Deaconess Medical Centre. We do not know whether results hold
> at Indian hospitals where TB co-infection is common.
>
> Second, this is a binary task. Real clinical X-ray reading involves
> 14 or more different findings, not just Normal versus Pneumonia.
>
> Third, 15 percent of WBC values were missing and imputed with the
> training set median — a limitation that real deployment would need to handle.
>
> All MIMIC-CXR data handling complied with the PhysioNet Data Use Agreement
> and CITI ethics training. No patient data is shared with this report."

---

## 🎬 SLIDE 13 — Next Phase: Indian Data (55 sec)

*[Trace the roadmap from top to bottom, then point to the right box.]*

> "Part 2 is done. AUC 0.971 on MIMIC-CXR. But the real test of any
> medical model is whether it works outside the data it was trained on.
>
> For the next phase, we plan to collect chest X-ray images paired with
> WBC counts from Indian hospitals and evaluate the Phase 3c pipeline on them.
>
> Why India specifically? The disease profile is very different from a US hospital.
> TB co-infection changes how pneumonia appears on an X-ray. Malnutrition
> alters how the immune system responds, which shifts WBC patterns.
> Scanner quality and report writing style also vary significantly.
>
> If the model holds up on Indian data, we can argue it has genuine clinical
> value for resource-limited settings — places where waiting 4 hours for
> a full lab panel is simply not realistic.
>
> If it does not hold up, we fine-tune it on Indian data and try again.
> Either way, that is where the project goes next."

---

## 🎬 SLIDE 14 — Full Project Journey (40 sec)

*[Trace the timeline bar from left to right.]*

> "Looking at the full project from the beginning, three lessons stand out.
>
> First: domain matters. ResNet pretrained on ImageNet fails on clinical
> chest X-rays. DenseNet pretrained on half a million real X-rays works.
>
> Second: the anti-leakage rule is non-negotiable. Stripping IMPRESSION
> from every report is what makes our numbers honest.
>
> Third: one well-chosen biomarker beats many poorly-available tests.
> WBC from a routine CBC, available in 15 minutes, closes 92 percent
> of the gap that 17 complex lab tests close."

---

## 🎬 SLIDE 15 — Conclusion & Thank You (45 sec)

*[Point to the final results table, then to the 5 takeaway points one by one.]*

> "To summarise Part 2 of PneumoFusionNet.
>
> We started at AUC 0.826 with just a chest X-ray.
> Adding leakage-free radiology report text pushed us to 0.949.
> Adding a single WBC count pushed us to 0.971 —
> sensitivity 92.3 percent, specificity 93.6 percent.
>
> The model only needs three things that are already available
> in any hospital within 30 minutes: the chest X-ray, the radiologist's
> FINDINGS and HISTORY, and a routine blood test.
>
> The next step is to bring this to Indian hospital data and test
> whether it holds up in a completely different clinical setting.
>
> Thank you for listening. The full code, notebooks, and results are
> available on GitHub at the link shown on screen."
