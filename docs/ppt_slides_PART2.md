# PneumoFusionNet — Part 2 Slide Deck Content
### IIT Guwahati | Trimester 8 Term Project | Ashutosh Yadav (23035010693)
### Format: Simple, Clean Academic Style (15 Slides, ~8-10 Minutes)

---

## Slide 1 — Title Slide
- **Title:** PneumoFusionNet: Part 2
- **Subtitle:** Image + Report Text + WBC Fusion for Pneumonia Detection
- **Author:** Ashutosh Yadav (Roll: 23035010693)
- **Programme:** B.Sc. (Hons.) Data Science & Artificial Intelligence
- **Institution:** Indian Institute of Technology Guwahati
- **Date:** September 2026

*Speaker note:* "Welcome to the Part 2 presentation of PneumoFusionNet. In Part 1, we established an image-only baseline on MIMIC-CXR and found that the image alone plateaued at AUC 0.826. In this part, I will present how adding radiology text and a single blood test—the WBC count—resolves diagnostic ambiguity and achieves 0.971 AUC."

---

## Slide 2 — Where Part 1 Left Off
- **What Part 1 Achieved:**
  - Architecture: DenseNet-121 + CBAM attention
  - Dataset: 1,989 MIMIC-CXR PA view radiographs
  - Results: AUC 0.826, Sensitivity 71.0%, Accuracy 76.3%
- **Why It Plateaued:**
  - Chest radiographs reveal pulmonary opacities, but cannot reflect the systemic immune response.
  - Opacities can indicate pneumonia, atelectasis, or heart failure fluid.
  - Clinical doctors rely on symptoms and routine labs to differentiate these causes.
- **Part 2 Roadmap:**
  - Phase 1: Image-only baseline (AUC 0.826)
  - Phase 2: Adding leakage-free radiology report text (AUC 0.949)
  - Phase 3c: Adding WBC count from routine blood tests (AUC 0.971)
  - Next Phase: Validation on Indian clinical cohorts

*Speaker note:* "Part 1 showed that tuning vision architectures on chest X-rays reaches a ceiling. An image cannot capture fever, illness duration, or systemic infection markers. Part 2 systematically adds two clinical sources available in emergency departments: the radiology report and routine WBC."

---

## Slide 3 — The Anti-Leakage Rule (No IMPRESSION)
- **Structure of a Radiology Report:**
  - `FINDINGS`: Visual observations made by the radiologist (e.g., opacity in right lower lobe).
  - `HISTORY`: Patient symptoms and reason for visit (e.g., fever for 3 days, cough).
  - `IMPRESSION`: The final diagnosis (e.g., "Pneumonia in right lower lobe").
- **The Label Leakage Problem:**
  - If a model reads the `IMPRESSION` section, it simply copies the radiologist's conclusion instead of learning to diagnose.
  - Many multimodal studies mistakenly leave this section in, producing artificially inflated metrics.
- **Our Solution:**
  - Programmatically strip the `IMPRESSION` section from all reports.
  - Use only `FINDINGS` + `HISTORY`, reflecting the clinical information available during active diagnostic workup.

*Speaker note:* "A critical design constraint was preventing label leakage. Every radiology report contains an IMPRESSION line which is the final diagnosis. We strip this section completely. Our model sees only observations and clinical history, forcing it to reason rather than memorize."

---

## Slide 4 — Phase 2: Adding the Radiology Report
- **Phase 2v1 (Initial Concat Baseline):**
  - Concatenated DenseNet image features with ClinicalBERT [CLS] token.
  - AUC: 0.911, Sensitivity: 80.6%.
  - Limitation: High specificity but poor sensitivity on borderline cases.
- **Phase 2v2 (Cross-Attention Architecture):**
  - Replaced [CLS] concatenation with 8-head cross-attention.
  - Image features query word-level token embeddings from the report.
  - Unfroze the top 2 layers of Bio_ClinicalBERT (learning rate 1e-5).
  - Applied Focal Loss (gamma = 2.0) and embedding Mixup (alpha = 0.2).
  - Result: AUC 0.949, Sensitivity 91.4% (catches 9 in 10 pneumonia cases).

*Speaker note:* "In Phase 2, we transitioned from simple feature concatenation to cross-attention. Instead of condensing the entire text into a single summary token, the image queries specific words in the text. This boosted sensitivity from 80% to 91%."

---

## Slide 5 — How Cross-Attention Works
- **Mechanism:**
  - **Query (Q):** 1024-dimensional visual feature vector from DenseNet-121.
  - **Key (K) & Value (V):** 768-dimensional token embeddings across all report words from ClinicalBERT.
  - **Attention Score:** Softmax((Q * K^T) / sqrt(64)) applied to Values.
- **Why This Helps Clinically:**
  - The model computes: 'Which words in the report correspond to the visual findings in this specific X-ray?'
  - Pathological terms like 'consolidation' or 'infiltrate' receive higher attention weights.
  - Yields a focused 512-dimensional text summary directly anchored to the radiograph.

*Speaker note:* "Cross-attention aligns what is seen with what was written. The image features act as a query into the report words. Words describing abnormalities receive high attention weights, creating a 512-dimensional summary relevant to the specific radiograph."

---

## Slide 6 — Phase 2 Results
- **Comparison Table (MIMIC-CXR Test Set, N = 565):**
  | Model | AUC | Sensitivity | Specificity | Accuracy |
  | :--- | :---: | :---: | :---: | :---: |
  | Phase 1 (X-ray only) | 0.826 | 71.0% | — | 76.3% |
  | Phase 2v1 (Concat) | 0.911 | 80.6% | 89.4% | 85.3% |
  | Phase 2v2 (Cross-Attention) | 0.949 | 91.4% | 86.3% | 88.6% |
  | Scale-Up Cohort (3,763 images) | 0.946 | 90.3% | 89.1% | 87.8% |
- **Key Takeaways:**
  - Adding leakage-free text provides a +12.3% AUC improvement over image alone.
  - Performance remains consistent when scaled up to 3,763 images (AUC drops by only 0.003), confirming lack of overfitting.

*Speaker note:* "Text alone adds over 12 percentage points in AUC and increases sensitivity from 71% to 91%. When tested on a doubled scale-up cohort of 3,763 images, the AUC was 0.946, confirming robust generalization."

---

## Slide 7 — Phase 3c: Adding WBC Count (Main Architecture)
- **Three Parallel Branches:**
  1. **Image Branch:** DenseNet-121 + CBAM -> 1024-d feature vector (frozen from Phase 1).
  2. **Report Branch:** Bio_ClinicalBERT -> 8-head cross-attention -> 512-d context vector.
  3. **WBC Branch:** WBC count from routine CBC -> 3-layer MLP (1 -> 128 -> 128 -> 64) -> 64-d embedding.
- **Fusion & Classification:**
  - Concatenation: z = [v_image (1024) || c_cross (512) || e_wbc (64)] -> 1600-dimensional vector.
  - Classification Head: MLP (1600 -> 512 -> 128 -> 2).
  - Fine-tuned with separate learning rates: BERT at 1e-5, fusion head at 2e-4, WBC encoder at 1e-3.
- **Why Encode WBC with an MLP?**
  - A WBC of 12 vs. 25 K/uL represents a clinical shift from borderline to acute infection.
  - A small MLP learns non-linear decision boundaries that a raw scalar cannot capture.

*Speaker note:* "Phase 3c is our primary proposed model. It takes the image features, attended report text, and a 64-dimensional learned embedding of the WBC count, concatenating them into a 1600-dimensional vector. We pass WBC through a small MLP so it can learn non-linear clinical thresholds."

---

## Slide 8 — Phase 3c Results
- **Primary Performance Metrics (Scale-Up Test Set, N = 565):**
  - **AUC:** 0.971
  - **Sensitivity:** 92.3%
  - **Specificity:** 93.6%
  - **Accuracy:** 92.9%
- **Threshold Setting Analysis:**
  | Threshold Strategy | Accuracy | Sensitivity | Specificity | Clinical Utility |
  | :--- | :---: | :---: | :---: | :--- |
  | Default (0.50) | 92.4% | 94.7% | 90.0% | Maximum sensitivity for screening |
  | Youden-J (0.559) | 92.9% | 92.3% | 93.6% | Best balanced operating point |
  | High Specificity (0.575) | 92.7% | 91.2% | 94.3% | Reduces false positives |
- **Gain Over Text-Only:** +2.5% AUC, +5.7% Sensitivity, +4.5% Specificity.

*Speaker note:* "Phase 3c achieves an AUC of 0.971. At the balanced threshold, sensitivity is 92.3% and specificity is 93.6%. At the default threshold, sensitivity reaches 94.7%, which is well-suited for emergency triage where missing sick patients carries high clinical risk."

---

## Slide 9 — Ablation Study
- **Systematic Modality Comparison (Identical Test Cohort N = 565):**
  | Model | Modalities Included | AUC | Sensitivity | Specificity | Accuracy |
  | :--- | :--- | :---: | :---: | :---: | :---: |
  | Phase 1 | X-ray only | 0.826 | 71.0% | — | 76.3% |
  | Phase 2 | X-ray + Report text | 0.946 | 86.6% | 89.1% | 87.8% |
  | Phase 3c | X-ray + Text + WBC | 0.971 | 92.3% | 93.6% | 92.9% |
- **Summary of Modality Contributions:**
  - Adding Report Text: +12.0% AUC, +15.6% Sensitivity.
  - Adding WBC Count: +2.5% AUC, +5.7% Sensitivity, +4.5% Specificity.
  - Overall Gain: +14.5% AUC and +21.3% Sensitivity over the image baseline.

*Speaker note:* "The ablation table highlights that each input earns its place. The text provides the largest single jump, while the single WBC value adds a crucial 5.7% gain in sensitivity. Together, they take performance from 0.826 to 0.971."

---

## Slide 10 — WBC-Only vs. Full 17-Feature Panel
- **Comparison Table:**
  | Model | Clinical Data Required | Turnaround Time | AUC | Sensitivity | Specificity |
  | :--- | :--- | :---: | :---: | :---: | :---: |
  | Phase 2v2 | None (text only) | Immediate | 0.946 | 86.6% | 89.1% |
  | Phase 3c | WBC only (1 test) | 15 minutes | 0.971 | 92.3% | 93.6% |
  | Phase 3 Full | 17 lab and vital features | 1-4 hours | 0.989 | 89.1% | 94.3% |
- **Key Clinical Findings:**
  - Phase 3c recovers **92.5%** of the full model's AUC gain using only 1 feature.
  - **Higher Sensitivity:** Phase 3c achieves 92.3% vs. 89.1% for the full model, missing fewer actual pneumonia cases.
  - **Turnaround:** Complete Blood Count (CBC) is available in 15 minutes; a 17-feature panel requires several hours of lab processing.

*Speaker note:* "Comparing Phase 3c to the full 17-feature model reveals our most practical finding: the full model reaches 0.989 AUC, but Phase 3c reaches 0.971 with higher sensitivity—92.3% vs. 89.1%. Moreover, WBC is ready in 15 minutes, whereas comprehensive lab panels take hours."

---

## Slide 11 — Clinical Logic Behind the WBC Gain
- **Diagnostic Complementarity:**
  - **Chest X-Ray:** Shows visual anatomical changes (e.g., opacity in right lower lobe). Remains ambiguous (pneumonia vs. atelectasis vs. fluid).
  - **WBC Count:** Reflects acute systemic immune activation (WBC > 11,000 / uL).
  - **Combined Decision:** Opacity + Elevated WBC strongly indicates infectious pneumonia rather than heart failure or lung collapse.
- **Data-Driven Decision Making:**
  - The model learns this clinical correlation directly from data without hand-crafted heuristic rules.

*Speaker note:* "The clinical rationale is straightforward: an X-ray shows the physical shadow, while WBC confirms whether the body's immune system is actively fighting an infection. The network learns this relationship directly from data."

---

## Slide 12 — Limitations
- **Single-Center Cohort:**
  - Dataset is derived entirely from Beth Israel Deaconess Medical Center (Boston, USA).
  - May not directly generalize to institutions with differing clinical demographics.
- **Binary Diagnostic Scope:**
  - Formulated as Normal vs. Pneumonia; real emergency chest radiographs frequently exhibit multiple co-occurring findings.
- **Missing Value Imputation:**
  - Approximately 15% of WBC records were missing and imputed using training-set medians.
- **Compliance & Ethics:**
  - All MIMIC-CXR usage complied with PhysioNet Data Use Agreements and CITI human research certification.

*Speaker note:* "We must acknowledge honest limitations. All data originated from one US academic medical center, the task is strictly binary, and 15% of WBC values required median imputation. Data handling adhered strictly to PhysioNet CITI ethical guidelines."

---

## Slide 13 — Next Phase: Validation on Indian Hospital Data
- **Planned Extension (Phase 4):**
  - Collect paired PA chest radiographs and routine CBC data from Indian hospital settings.
  - Evaluate and fine-tune the Phase 3c pipeline on Indian patient cohorts.
- **Why Indian Cohort Validation Matters:**
  - **Disease Context:** High prevalence of Tuberculosis (TB) co-infection alters radiological presentation.
  - **Immune Variation:** Malnutrition and endemic factors alter baseline WBC distributions.
  - **Hardware Diversity:** Greater variability in X-ray equipment (portable units, computed radiography).
  - **Clinical Impact:** A lightweight model needing only an X-ray and CBC is practical for resource-constrained clinics.

*Speaker note:* "The logical next step is validating this framework on Indian hospital data. Differences in disease profile—notably tuberculosis co-infection, nutritional status, and hardware variability—make Indian clinical validation essential."

---

## Slide 14 — Complete Project Journey & Lessons Learned
- **Journey Overview:**
  - Stage 0 (Kaggle / IU): Public dataset architecture validation (AUC 0.97).
  - Stage 1 (MIMIC Pilot): First clinical test on 139 cases (AUC 0.85).
  - Stage 2 (MIMIC Scale-Up): ResNet failure at scale -> switched to DenseNet-121 + CBAM (AUC 0.826).
  - Phase 2 (Text Integration): Cross-attention with Bio_ClinicalBERT (AUC 0.949).
  - Phase 3c (Minimal Fusion): Image + Text + WBC fusion (AUC 0.971).
- **Core Lessons:**
  1. **Domain pretraining matters:** ImageNet weights fail on clinical chest X-rays; medical pretraining is essential.
  2. **Anti-leakage is non-negotiable:** Reports must be cleaned of diagnosis sections.
  3. **One targeted biomarker suffices:** A single rapid lab test can match complex multi-feature panels.

*Speaker note:* "Reflecting on the complete project, three lessons stand out: domain pretraining is essential for medical imaging, anti-leakage filters are required for honest evaluation, and a single well-chosen lab value can rival a complex lab panel."

---

## Slide 15 — Summary & Conclusion
- **Final Summary Table:**
  | Stage | Input Modalities | Test AUC | Sensitivity | Specificity |
  | :--- | :--- | :---: | :---: | :---: |
  | Baseline | X-ray only | 0.826 | 71.0% | — |
  | Phase 2 | + Report text (leakage-free) | 0.949 | 91.4% | 86.3% |
  | Phase 3c | + WBC count (routine blood test) | **0.971** | **92.3%** | **93.6%** |
  | **Overall Gain** | | **+14.5%** | **+21.3%** | — |
- **Concluding Remarks:**
  - High diagnostic performance does not require dozens of laboratory tests.
  - Three routine inputs available within 30 minutes suffice for high-accuracy triage.
  - Code and models are available on GitHub.

*Speaker note:* "To conclude, combining an X-ray, leakage-free report text, and a routine WBC count brings pneumonia detection performance from 0.826 to 0.971 AUC while maintaining 92% sensitivity. The solution is efficient and directly applicable to clinical workflows. Thank you."
