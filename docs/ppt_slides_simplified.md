# 🎞️ PneumoFusionNet — Part 1 PPT Slide Content (Storyline Version)
### For Google Slides / PowerPoint | 6–8 min | 9 Slides

---

## SLIDE 1 — Title Slide

```
  🫁  PneumoFusionNet
      Part 1: The Journey to Multimodal Pneumonia Detection

  Ashutosh Yadav  |  IIT Guwahati
```
*Speaker note:* "Welcome to Part 1 of PneumoFusionNet. I'll walk you through our research journey—starting from the core problem, our initial experiments on public datasets, our search for the perfect dataset, and what happened when we ran our pilot architecture on real clinical data."

---

## SLIDE 2 — The Problem: Ambiguity & The Leakage Trap

**1. Visual Ambiguity (Image Problem)**
> "The same cloudy patch in a lung can look like pneumonia, fluid buildup, or a collapsed lung section." - *Project Report*
> Doctors use clinical context to decide. Image-only AI is forced to guess.

**2. Label Leakage (The Multimodal Trap)**
> Adding radiology reports helps, but the last section (`IMPRESSION`) contains the doctor's final written diagnosis. If a model reads this, it simply memorises the answer instead of learning to diagnose.

---

## SLIDE 3 — The Inspiration

**Research Inspiration:**
> S. Yu et al., *"A multi-modal deep learning solution for precise pneumonia diagnosis: the PneumoFusion-Net model,"* Frontiers in Physiology, March 2025.

**Our Goal:**
Build a multimodal model inspired by this research, combining X-ray images, clinical data, and radiological reports—while strictly avoiding label leakage.

---

## SLIDE 4 — Initial Experiments: Kaggle Chest X-Ray

**Our Starting Point:**
We built a pilot architecture (ResNet50 + DSC + GCSA) and tested it on the open-source Kaggle Chest X-Ray dataset.

**Findings (`v2_Enhanced_Image_Model`):**
> Achieved **92% Accuracy**.
> The architecture was validated, but Kaggle is pre-cleaned and only contains images. We needed text.

---

## SLIDE 5 — The Dataset Search & The Issue

**Testing Indiana University (IU) Dataset:**
> We tried the IU Dataset (`v3-1-iu-dataset-multimodal`) to test our multimodal pipeline.

**The Issue:**
> Public datasets are too clean or lack paired data. To build a *perfect multimodal model*, we needed a dataset containing:
> 1. X-Ray Images
> 2. Clinical Data (Symptoms, History)
> 3. Unstructured Radiological Reports (Findings)

---

## SLIDE 6 — Securing the MIMIC-CXR Dataset

**The Solution: MIMIC-CXR**
We identified the MIMIC-CXR hospital dataset as the perfect match.

**The Access Challenge:**
> MIMIC-CXR is highly restricted.
> We completed CITI human subjects training and signed a Data Use Agreement (DUA) with PhysioNet to gain authorised access.

---

## SLIDE 7 — The Pilot Test

**Testing on a Small Subset:**
> Due to the massive size of the MIMIC dataset, we started with a small pilot dataset to verify our pipeline and anti-leakage filters.

**The Results:**
> We achieved very good results during the pilot phase.
> *However*, because the dataset size was extremely small, the results were not statistically reliable. We needed to scale up.

---

## SLIDE 8 — The Scale-Up Failure

**Testing the Pilot Architecture at Scale:**
We upscaled the dataset to nearly 2,000 images and ran the exact same pilot architecture.
(`Phase-1.1-pilot_arch_scaleUp_ResNet_DSC_GCSA.ipynb`)

**The Result:**
> **FAILED.** (AUC dropped to 0.713, Accuracy 66%).
> ImageNet-pretrained ResNet50 failed to learn the complex pathology of real clinical X-rays at scale.

---

## SLIDE 9 — Conclusion & Next Steps

**Conclusion of Part 1:**
1. We successfully built a multimodal, anti-leakage data pipeline.
2. We proved that public dataset results (Kaggle) do not translate to real clinical environments.
3. Our pilot architecture completely failed at clinical scale.

**What's Next (The Solution):**
> We must discard the pilot architecture.
> To solve the scale-up failure, we have started working on a completely different model architecture using **DenseNet** (which has domain-specific X-ray pretraining) for the final multimodal fusion.
