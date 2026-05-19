# ResNet-GCSA-Pneumonia
This repository implements the image-processing module of the PneumoFusion-Net framework, as described in the 2025 Frontiers in Physiology study. The model leverages a modified ResNet50 backbone integrated with Global Channel-Spatial Attention (GCSA) and Depthwise Separable (DSC)  to achieve high-accuracy pneumonia classification from chest X-ray/




# 🫁 PneumoFusionNet for Pneumonia Detection

🚀 Current Version: **v2 — Enhanced Image Model (GCSA + DSC + Grad-CAM + Full Evaluation)**

---

## 📌 Overview
This project implements a deep learning model for detecting pneumonia from chest X-ray images.  
The model improves a standard CNN pipeline by integrating advanced techniques such as attention mechanisms and efficient convolutions.

---

## 🧠 Key Features

✅ Modified ResNet50 (adapted for grayscale images)  
✅ Depthwise Separable Convolution (DSC)  
✅ Global Channel Spatial Attention (GCSA)  
✅ Overfitting reduction techniques (dropout, freezing)  
✅ Full evaluation metrics (Precision, Recall, F1-score, ROC-AUC)  
✅ Grad-CAM visualization (Explainable AI)

---

## 🆕 Version Progress

### ✅ v1 – Basic Image Model
- Standard CNN / ResNet
- Basic training pipeline
- Limited evaluation

---

### ✅ v2 – Enhanced Image Model (Latest 🔥)
- Modified ResNet50 (1-channel input)
- Added GCSA Attention
- Added Depthwise Separable Convolution (DSC)
- Improved training (reduced overfitting)
- Full evaluation metrics:
  - Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC
- Grad-CAM visualization (Explainable AI)

---

## 🧠 Model Architecture
