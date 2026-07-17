# 🎬 PneumoFusionNet Part 1 — Video Script
### Narrated Walkthrough | Target: 5 Minutes | 9 Slides

---

## SEGMENT 1 — Hook & Overview (Slide 1)

**[SPEAK]:**
> "Welcome to Part 1 of PneumoFusionNet. My name is Ashutosh Yadav from IIT Guwahati. Today I'll walk you through our research journey—starting from defining the core clinical problem, testing on public datasets, finding the perfect clinical data, and what happened when we ran our pilot architecture on real medical records."

---

## SEGMENT 2 — The Problem (Slide 2)

**[SPEAK]:**
> "Our project tackles two major challenges. First is visual ambiguity. A cloudy patch on a chest X-ray can look like pneumonia, fluid buildup, or a collapsed lung. Doctors don't just look at the image; they use clinical context. If an AI only has the image, it's forced to guess.
>
> To solve this, we want to add radiology reports. But that introduces the second challenge: label leakage. Every report ends with an 'IMPRESSION' section where the doctor writes the final diagnosis. If the AI reads this, it simply memorises the answer instead of learning how to diagnose. We have to prevent this."

---

## SEGMENT 3 — The Inspiration (Slide 3)

**[SPEAK]:**
> "Our work is deeply inspired by a March 2025 paper by S. Yu and colleagues, titled 'A multi-modal deep learning solution for precise pneumonia diagnosis: the PneumoFusion-Net model,' published in Frontiers in Physiology. 
>
> Our goal was to build a model inspired by this research, combining X-ray images, clinical data, and radiological reports, while strictly avoiding the label leakage trap."

---

## SEGMENT 4 — Initial Experiments on Kaggle (Slide 4)

**[SPEAK]:**
> "We started our journey with the Kaggle Chest X-Ray dataset. We built a pilot image architecture using a ResNet50 backbone, enhanced with Depthwise Separable Convolutions (DSC) and Global Context Spatial Attention (GCSA).
>
> As shown in our 'v2 Enhanced Image Model' notebook, this achieved 92% accuracy. The architecture was validated. However, Kaggle data is pre-cleaned and only contains images. To achieve our multimodal goal, we needed text."

---

## SEGMENT 5 — The Dataset Search & The Issue (Slide 5)

**[SPEAK]:**
> "Next, we tested the Indiana University dataset to check our multimodal pipeline, as seen in our 'v3-1' notebook. 
>
> But we quickly ran into an issue. Public datasets are either too clean or lack properly paired multimodal data. To build the perfect multimodal model, we needed a single dataset containing three things: X-Ray images, clinical patient data like symptoms and history, and unstructured radiological findings."

---

## SEGMENT 6 — Securing MIMIC-CXR (Slide 6)

**[SPEAK]:**
> "We found the perfect match: the MIMIC-CXR hospital dataset. It contains all three modalities from real hospital patients.
>
> However, gaining access was a challenge because the data is highly restricted. We had to complete CITI human subjects training and sign a strict Data Use Agreement with PhysioNet. Once that was done, we finally received access."

---

## SEGMENT 7 — The Pilot Test (Slide 7)

**[SPEAK]:**
> "MIMIC-CXR is massive, so we started with a small pilot dataset. This allowed us to verify our pipeline and ensure our anti-leakage filters were correctly stripping the 'IMPRESSION' sections from the reports.
>
> We achieved very good results during this pilot phase. But because the dataset was so small, the results were statistically unreliable. We had to test it on a much larger scale to know if it actually worked."

---

## SEGMENT 8 — The Scale-Up Failure (Slide 8)

**[SPEAK]:**
> "This brings us to the final test of Part 1. We upscaled the dataset to nearly 2,000 images and ran the exact same pilot architecture—ResNet50 plus DSC plus GCSA. You can see this in our 'Phase 1.1 scale-up' notebook.
>
> The result? It failed completely. The AUC dropped to 0.713, and accuracy fell to 66%. 
> 
> Why? Because ResNet50 is pretrained on ImageNet—pictures of cats and dogs. It simply failed to learn the complex, messy pathology of real clinical X-rays at scale."

---

## SEGMENT 9 — Conclusion & Next Steps (Slide 9)

**[SPEAK]:**
> "To conclude Part 1: We successfully built a multimodal, anti-leakage data pipeline. We proved that high accuracy on public datasets like Kaggle does not translate to real clinical environments. And we found that our pilot architecture completely fails at clinical scale.
>
> So, what's next? For the final stage of our project, we must discard the pilot architecture. To solve the scale-up failure, we have started working on a completely different model architecture using DenseNet, which has domain-specific X-ray pretraining, for our final multimodal fusion.
>
> Thank you."
