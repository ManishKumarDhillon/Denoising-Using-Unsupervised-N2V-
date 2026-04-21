Self supervised denoising and analysis of TEM/STEM images using Noise2Void (U-Net) with GMM-based atomic column classification.
# 🧪 TEM/STEM Image Denoising and Analysis using Noise2Void (U-Net)

This repository presents a deep learning pipeline for denoising, segmentation, and analysis of TEM/STEM images using a self-supervised approach. The method is based on Noise2Void (N2V) implemented with a U-Net architecture, enabling high-quality denoising without requiring clean ground truth data.

## 🚀 Overview

Low-dose TEM/STEM imaging often suffers from significant noise, making downstream analysis (such as atomic column detection) challenging. This project addresses this problem using:

-   Self-supervised denoising (Noise2Void)
-   Deep learning (U-Net architecture)
-   Unsupervised classification (GMM + BIC)

The pipeline is trained on simulated noisy data and tested on real experimental images.

## 🧠 Methodology

### 1. Denoising (N2V + U-Net)

-   No clean labels required
-   Learns pixel context for restoration

### 2. Analysis

-   Atomic column detection
-   Inter-atomic distance estimation

### 3. Classification

-   Gaussian Mixture Model (GMM)
-   Bayesian Information Criterion (BIC) for optimal clusters

## 📦 Outputs

-   Denoised images (PNG)
-   Model checkpoint (best.pth)
-   Clustering results

## 🛠️ Tech Stack

Python, PyTorch, NumPy, OpenCV, Scikit-learn, Matplotlib

## 📬 Contact

Manish Kumar\
kumar.216@alumni.iitj.ac.in
