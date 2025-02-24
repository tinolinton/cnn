import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
MODEL_PATH = "Custom_CNN_best_model.h5"

def f1_metric(y_true, y_pred):
    # Define F1 metric function if needed
    pass

model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'f1_metric': f1_metric})

# Initialize Flask app
app = Flask(__name__)

# Image preprocessing function
def preprocess_image(image, target_size=(150, 150)):
    img = load_img(image, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Define classification route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img_array = preprocess_image(file)
        prediction = model.predict(img_array)
        class_label = 'fake' if prediction[0] < 0.5 else 'real'
        confidence = 1 - prediction[0] if class_label == 'fake' else prediction[0]

        return jsonify({'class': class_label, 'confidence': float(confidence[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Default route
@app.route('/')
def home():
    return jsonify({'message': 'Image classification API is running'})

if __name__ == '__main__':
    app.run(debug=True)