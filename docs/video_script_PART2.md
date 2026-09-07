# PneumoFusionNet — Part 2 Video Presentation Script
### IIT Guwahati | Term Project (Trimester 8) | Ashutosh Yadav (23035010693)
### Estimated Duration: ~8–10 minutes (~35–45 sec per slide)

---

## Slide 1 — Title (45 sec)

*[Show title slide. Speak clearly and at a measured pace.]*

> "Hello, my name is Ashutosh Yadav, Roll Number 23035010693, from the B.Sc. Data Science and Artificial Intelligence programme at IIT Guwahati.
>
> This is Part 2 of my Term 8 project on PneumoFusionNet — a multimodal deep learning framework for pneumonia detection using chest X-rays, radiology reports, and routine blood tests.
>
> In Part 1, we observed that an image-only model plateaued at AUC 0.826. Today, I will walk you through why that ceiling exists, how we addressed it using multimodal clinical data, and how we reached an AUC of 0.971 with 92% sensitivity using inputs available within 30 minutes in an emergency setting."

---

## Slide 2 — Where Part 1 Left Off (45 sec)

*[Refer to the summary on the left and the progression roadmap on the right.]*

> "To quickly recap Part 1: we implemented a DenseNet-121 architecture with CBAM attention trained on 1,989 MIMIC-CXR chest radiographs. Our best performance reached an AUC of 0.826 with 71 percent sensitivity.
>
> The reason for this ceiling is clear: a chest radiograph shows structural opacities, but cannot reflect systemic infection or patient history. Shadows can represent pneumonia, atelectasis, or pulmonary edema.
>
> In Part 2, we systematically address this ambiguity by incorporating two complementary sources: the radiologist's written observations and a routine WBC blood count."

---

## Slide 3 — The Anti-Leakage Rule (50 sec)

*[Point to the sample report, distinguishing FINDINGS/HISTORY from IMPRESSION.]*

> "A crucial methodological requirement in our text processing pipeline is the prevention of label leakage.
>
> Clinical radiology reports contain three primary sections: FINDINGS, which details anatomical observations; HISTORY, which provides clinical symptoms; and IMPRESSION, which gives the radiologist's final diagnosis.
>
> If a language model is trained on reports containing the IMPRESSION section, it merely memorizes the diagnostic label rather than learning to correlate findings with disease.
>
> We programmatically strip the IMPRESSION section from every report. Our models only see FINDINGS and HISTORY, strictly mirroring the information available during active clinical assessment."

---

## Slide 4 — Phase 2: Adding the Radiology Report (50 sec)

*[Direct attention to the progression from Version 1 concatenation to Version 2 cross-attention.]*

> "In Phase 2, we integrated the leakage-free reports using Bio_ClinicalBERT.
>
> Our initial baseline, Version 1, concatenated image embeddings with the BERT CLS token, achieving an AUC of 0.911 but a modest sensitivity of 80.6%.
>
> In Version 2, we introduced seven technical refinements. Crucially, we replaced simple concatenation with an 8-head cross-attention module, allowing the image features to attend directly to word tokens in the text. We also unfroze the top two BERT layers, incorporated Focal Loss, and applied embedding Mixup.
>
> This improved sensitivity to 91.4% and AUC to 0.949."

---

## Slide 5 — How Cross-Attention Works (45 sec)

*[Explain the Query-Key-Value flow depicted on the slide.]*

> "Here is how our cross-attention mechanism functions:
>
> The 1024-dimensional image features serve as the Query, while the 768-dimensional ClinicalBERT token representations act as Keys and Values.
>
> By computing scaled dot-product attention, the model determines which specific words in the report relate to the visual patterns observed in the radiograph. Words describing focal abnormalities receive higher attention weight, generating a 512-dimensional text summary directly anchored to visual findings."

---

## Slide 6 — Phase 2 Results (40 sec)

*[Review the metrics in the table and point to the scale-up findings.]*

> "Reviewing the Phase 2 metrics: adding the radiology report yields a 12.3 percentage point gain in AUC over image alone, while sensitivity increases from 71.0% to 91.4%.
>
> When evaluated on our expanded scale-up cohort of 3,763 radiographs, performance remained remarkably steady at 0.946 AUC and 90.3% sensitivity. This demonstrates that the model is learning generalizable multimodal associations rather than overfitting."

---

## Slide 7 — Phase 3c: Adding WBC Count (55 sec)

*[Walk through the three input branches and the fusion mechanism.]*

> "Phase 3c represents our core proposed model: minimal triple fusion.
>
> We retain the DenseNet image branch and the ClinicalBERT text branch, and introduce a third branch: the White Blood Cell count from a routine Complete Blood Count, or CBC.
>
> Rather than passing WBC as a raw scalar, we pass it through a 3-layer MLP to produce a 64-dimensional embedding. This enables the network to learn non-linear clinical thresholds—distinguishing normal counts from borderline and severe elevations.
>
> The three representations concatenate into a 1600-dimensional vector and pass to a final classification head."

---

## Slide 8 — Phase 3c Results (45 sec)

*[Highlight the four key metrics and discuss threshold flexibility.]*

> "On the scale-up test cohort of 565 patients, Phase 3c achieved an AUC of 0.971, with 92.3% sensitivity, 93.6% specificity, and 92.9% accuracy.
>
> Adding this single lab value provided an additional +2.5% AUC and +5.7% sensitivity over the text-augmented model.
>
> Depending on clinical priorities, the threshold can be adjusted: the default 0.50 threshold yields 94.7% sensitivity, which is ideal for emergency screening where minimizing false negatives is paramount."

---

## Slide 9 — Ablation Study (40 sec)

*[Step through the ablation rows to summarize the progression.]*

> "Our ablation study confirms the incremental contribution of each clinical modality.
>
> X-ray alone yields an AUC of 0.826. Adding the radiology report lifts performance to 0.946. Introducing the single WBC count further elevates the AUC to 0.971.
>
> Overall, the progression from an image-only baseline to Phase 3c delivers a total gain of +14.5% in AUC and +21.3% in sensitivity."

---

## Slide 10 — WBC-Only vs. Full 17-Feature Panel (50 sec)

*[Contrast Phase 3c with the full 17-feature model in terms of both metrics and turnaround time.]*

> "A particularly important finding emerges when comparing Phase 3c against our comprehensive Phase 3 model, which uses 17 lab and vital signs.
>
> While the full model reaches an AUC of 0.989, Phase 3c achieves 0.971, recovering 92.5% of that performance gain with just one biomarker.
>
> More importantly, Phase 3c achieves higher sensitivity—92.3% compared to 89.1%—missing fewer actual pneumonia cases. Practically, a CBC test takes 15 minutes, whereas a 17-feature panel requires several hours of laboratory processing."

---

## Slide 11 — Clinical Logic Behind the WBC Gain (45 sec)

*[Explain the diagnostic combination of radiographic and laboratory findings.]*

> "The clinical rationale behind this improvement is clear.
>
> A pulmonary opacity on a chest X-ray is visually ambiguous—it could indicate pneumonia, atelectasis, or cardiogenic edema.
>
> An elevated WBC count above 11,000 per microliter indicates an active systemic immune response. When combined with a radiographic opacity, it strongly favors an infectious etiology. The network learns this diagnostic reasoning directly from patient data."

---

## Slide 12 — Limitations (40 sec)

*[Walk through the four limitation points objectively.]*

> "We must note several methodological limitations.
>
> First, our dataset originates from a single institution, Beth Israel Deaconess Medical Center in Boston, which may not capture international demographic diversity.
>
> Second, the task is formulated as binary classification, whereas real-world clinical interpretation involves multiple concurrent findings.
>
> Third, roughly 15% of WBC values required median imputation.
>
> All experiments were conducted under approved PhysioNet Data Use Agreements and CITI research ethics training."

---

## Slide 13 — Next Phase: Validation on Indian Hospital Data (50 sec)

*[Explain the planned future work in Indian clinical settings.]*

> "For the next phase of this research, we plan to validate and fine-tune Phase 3c on chest radiographs and CBC data collected from Indian hospitals.
>
> In the Indian clinical context, epidemiological differences such as endemic tuberculosis alter radiographic presentation, while nutritional variation influences baseline immune markers.
>
> Validating the model across varied portable and computed radiography systems will provide an honest test of its utility in resource-constrained community healthcare."

---

## Slide 14 — Complete Project Journey (40 sec)

*[Trace the timeline from Stage 0 to Phase 3c.]*

> "Looking across the complete project lifecycle:
>
> We progressed from initial public dataset experiments to clinical scale-up, identified the failure of generic ImageNet models, and rebuilt our foundation using domain-specific DenseNet backbones, cross-attention, and targeted lab fusion.
>
> The key lessons: domain pretraining is essential, anti-leakage data preparation is mandatory, and a single targeted biomarker can achieve comparable utility to extensive lab panels."

---

## Slide 15 — Summary & Conclusion (40 sec)

*[Conclude on the final summary table and direct viewers to the GitHub repository.]*

> "In summary, PneumoFusionNet Part 2 demonstrates that combining chest X-rays with leakage-free reports and a single WBC count elevates pneumonia detection from 0.826 to 0.971 AUC with 92.3% sensitivity.
>
> The framework requires only routine tests available within 30 minutes of emergency admission, providing a practical tool for clinical triage.
>
> The project notebooks and reproducible code are available on GitHub. Thank you."
