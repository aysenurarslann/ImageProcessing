# Fruit Ripeness Detection Using Hyperspectral Imaging

This project aims to determine the ripeness of kiwi fruits using hyperspectral imaging technology. Unlike traditional methods, it offers a fast, non-invasive, and accurate approach for evaluating fruit quality.

## 🔍 Project Overview

- **Dataset**: 1,916 hyperspectral images (.hdr and .bin), each with 219x240 resolution and 224 spectral bands.
- **Objective**: Classify the ripeness level of kiwi fruits based on spectral data.
- **Band Selection**: Competitive Adaptive Reweighted Sampling (CARS) algorithm was chosen.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied.
- **Models Used**: 
  - Deep Learning: AlexNet
  - Machine Learning: Random Forest, Decision Tree

## 🧪 Technologies Used

- Python
- Libraries: NumPy, Pandas, Matplotlib, Scikit-learn, Spectral Python (SPy), skimage
- Deep Learning Frameworks: Keras, TensorFlow

## ⚙️ Hardware Configuration

- **CPU**: Intel Core i7
- **RAM**: 32 GB
- **GPU**: NVIDIA GeForce RTX 2080
- **Hyperspectral Cameras**: FX 10, INNO-SPEC RedEye 1.7, Corning microHSI 410 Vis-NIR

## 📊 Model Performance Comparison

| Model         | Stage       | Accuracy | Precision | Recall | F1-Score |
|---------------|-------------|----------|-----------|--------|----------|
| AlexNet       | Before CARS | 0.4930   | 0.2431    | 0.4930 | 0.3256   |
| AlexNet       | After CARS  | 0.4931   | 0.2431    | 0.4931 | 0.3256   |
| Random Forest | Before CARS | 0.6388   | 0.6390    | 0.6389 | 0.6385   |
| Random Forest | After CARS  | 0.7881   | 0.7882    | 0.7881 | 0.7882   |
| Decision Tree | Before CARS | 0.4000   | 0.5000    | 0.6700 | 0.5700   |
| Decision Tree | After CARS  | 0.6000   | 0.6000    | 0.6000 | 0.6000   |

## 🔎 Key Findings

- **Best Performing Model**: Random Forest (after CARS)
- **Band Selection**: CARS significantly improved the performance of tree-based models.
- **AlexNet**: Showed poor performance; further optimization or alternative architectures are recommended.

## 🚀 Future Work

- Evaluate other deep learning architectures (e.g., ResNet, Inception)
- Apply the pipeline to different fruits or crops
- Optimize hyperparameters for all models
- Integrate additional feature extraction (texture, color) for improved prediction
