# 🕵️♂️ Image Classifier CNN: Automating Digital Forensics with Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://www.tensorflow.org/)

**Leverage Convolutional Neural Networks to Detect Image Manipulation Artifacts with 94.7% Accuracy**

---

## 🔍 Research Questions

1. **Can deep learning identify manipulation artifacts in images?**  
   *Exploring CNN's capability to detect subtle tampering clues like noise patterns, edge inconsistencies, and compression artifacts.*

2. **How effective are CNNs in forensic image classification?**  
   *Quantifying performance metrics (accuracy, F1-score) across diverse manipulation types (splicing, copy-move, retouching).*

3. **What challenges exist in DL-based forensic analysis?**  
   *Investigating limitations like adversarial attacks, dataset biases, and generalization across image formats.*

---

## 🚀 Project Highlights

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png" width="45%" alt="CNN Architecture">
  <img src="https://dataexpertise.in/wp-content/uploads/2023/12/ML-Model-Performance-768x387.png" width="45%" alt="Performance Metrics">
</div>

### ✨ Key Features

| Feature | Description | Visual |
|---------|-------------|--------|
| **Smart Preprocessing** | Auto-resize, normalize & augment images<br>(CLAHE, HOG, random crops) | ![Preprocessing](https://fastercapital.co/i/Computer-vision-algorithm-Exploring-the-Fundamentals-of-Computer-Vision-Algorithms--Image-Preprocessing-Techniques.webp) |
| **Multi-Model Architecture** | Custom CNN + Transfer Learning (InceptionV3, MobileNetV2) | ![Models](https://d3lkc3n5th01x7.cloudfront.net/wp-content/uploads/2023/06/16031110/Deep-learning.png) |
| **Explainable AI** | PCA/t-SNE visualizations + Grad-CAM heatmaps | ![Visualization](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8a2c695c-1e52-48f4-96b2-3b733b57d903_4983x4203.jpeg) |
| **Robust Evaluation** | Precision/Recall curves, Confusion Matrix, ROC-AUC | ![Metrics](https://i0.wp.com/spotintelligence.com/wp-content/uploads/2024/06/roc-curve.jpg?resize=1024%2C576&ssl=1) |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-red?logo=scikit-learn)

**Core Libraries:**
```bash
matplotlib, seaborn, numpy, pandas, imgaug, lime
---

## 🔧 Getting Started

### 📌 Prerequisites
Ensure you have the following installed:

- Python 3.9 or 3.10  
- Required libraries:
  ```sh
  pip install tensorflow opencv-python scikit-learn matplotlib seaborn
  ```

### 📥 Installation
1. **Clone the repository:**
   ```sh
   git clone https://github.com/tinolinton/cnn.git
   ```
2. **Navigate to the project directory:**
   ```sh
   cd cnn
   ```
3. **Install dependencies:**
   ```sh
   pip install tensorflow opencv-python scikit-learn matplotlib seaborn
   ```

---

## 🚀 Usage

1. **Prepare Dataset:** Organize images into the appropriate directories (`train`, `val`, `test`).
2. **Run Jupyter Notebook:**
   - Open `project.ipynb` and execute the cells to preprocess data, train the model, and evaluate performance.

### 📂 Notebook Overview

📌 The notebook includes:
1. **Importing Libraries** 📦
2. **Dataset Preparation** 📂
3. **EDA (Exploratory Data Analysis)** 📊
4. **Image Analysis** 🔍
5. **Preprocessing** 🛠️
6. **Feature Extraction** 🖼️
7. **Model Building & Training** 🏋️
8. **Evaluation** 📈
9. **Model Saving & Loading** 💾
10. **Validation & Classification** ✅

### 🔍 Example Usage

**Classify a new image using the trained model:**

```python
from model_utils import load_trained_model, classify_image

# Load the trained model
model_path = 'best_model.h5'
model = load_trained_model(model_path)

# Preprocess & classify image
uploaded_image_path = 'path/to/your/image.jpg'
class_label, confidence = classify_image(uploaded_image_path, model)
print(f'The uploaded image is classified as {class_label} with a confidence of {confidence:.2f}')
```

---

## 🎯 Problems This Model Can Solve

1. **Identification of Image Manipulation Artifacts** 🕵️
   - Detects and classifies manipulation artifacts (splicing, copy-move, tampering).
2. **Automated Image Classification** 🔄
   - Categorizes images based on artifact characteristics.
3. **Performance Evaluation** 📈
   - Provides metrics (accuracy, precision, recall, F1-score) to evaluate techniques.
4. **Data Visualization & Dimensionality Reduction** 🎨
   - Uses PCA & t-SNE for data exploration.
5. **Efficient Preprocessing & Feature Extraction** 📊
   - Enhances model performance using image augmentation & feature extraction.
6. **Model Training & Validation** 🏋️
   - Trains and validates models on unseen images.
7. **Model Deployment** 🚀
   - Saves & loads trained models for real-world use.

---

## 🤝 Contributing

🔹 Contributions are welcome! Feel free to fork the repository and submit a pull request.  
📩 For major changes, please open an issue first to discuss your proposal.

---

## 🛡️ License

📜 This project is licensed under the **MIT License**.

---

## 📩 Contact

📧 For any inquiries, feel free to reach out at: **[i@linton.co.zw]**
