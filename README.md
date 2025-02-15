# Image Classifier CNN

**Automating Digital Forensics Using Deep Learning-Based Image Classification**

---

## Research Questions

This project aims to address the following research questions:

1. **Can deep learning-based image classification techniques be used to identify manipulation artifacts in images?**
2. **How effective are these approaches in classifying images based on their manipulation artifact characteristics?**
3. **What are the limitations and challenges associated with using deep learning-based image classification techniques for digital forensics?**

---

## Project Overview

This project leverages deep learning techniques to automate the identification of image manipulation artifacts for digital forensic purposes. By using convolutional neural networks (CNNs), the model is trained to classify images based on their artifact characteristics.

![CNN Architecture](https://upload.wikimedia.org/wikipedia/commons/6/63/Typical_cnn.png)

---

## Features

- **Image Preprocessing**: Automatic resizing, normalization, and augmentation of images.
- **Deep Learning Model**: Implementation of CNNs for binary or multi-class image classification.
- **Visualization**: Use of PCA and t-SNE for dimensionality reduction and data visualization.
- **Performance Evaluation**: Metrics such as accuracy, precision, recall, and F1-score for model evaluation.

---

## Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.9 or 3.10
- Required libraries (install via `pip install tensorflow opencv-python scikit-learn matplotlib seaborn`)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/tinolinton/cnn.git
   ```
2. Navigate to the project directory:
   ```sh
   cd cnn
   ```
3. Install the required libraries:
   ```sh
   pip install tensorflow opencv-python scikit-learn matplotlib seaborn
   ```

### Usage

1. Prepare your dataset by organizing images into the appropriate directories (`train`, `val`, `test`).
2. Run the Jupyter notebook provided to preprocess the data, train the model, and evaluate performance:
   - [project.ipynb](http://_vscodecontentref_/1)

### Notebook Overview

The notebook is structured as follows:

1. **Importing Libraries**: Import necessary libraries for data processing, visualization, and model building.
2. **Dataset Preparation**: Load and preprocess the dataset, including resizing, normalization, and augmentation.
3. **Exploratory Data Analysis (EDA)**: Visualize class distribution, sample images, and basic statistics.
4. **Image Analysis**: Analyze image size, aspect ratio, contrast, and brightness distributions.
5. **Preprocessing**: Apply custom preprocessing techniques, including histogram equalization and HOG feature extraction.
6. **Feature Extraction**: Extract features using PCA and t-SNE for dimensionality reduction.
7. **Model Building**: Define and train different CNN models (Custom CNN, InceptionV3, MobileNetV2).
8. **Evaluation**: Evaluate model performance using accuracy, loss, confusion matrix, and ROC curve.
9. **Saving and Loading Models**: Save the trained model and load it for future use.
10. **Validation**: Validate the model on new images and classify them.

### Example Usage

To classify a new image using the trained model:

1. Load the trained model:

   ```python
   model_path = 'best_model.h5'
   model = load_trained_model(model_path)
   ```

2. Preprocess and classify the image:
   ```python
   uploaded_image_path = 'path/to/your/image.jpg'
   class_label, confidence = classify_image(uploaded_image_path, model)
   print(f'The uploaded image is classified as {class_label} with a confidence of {confidence:.2f}')
   ```

### Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

### License

This project is licensed under the MIT License.

---

## Contact

For any inquiries, please contact [i@linton.co.zw].
