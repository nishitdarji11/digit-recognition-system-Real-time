# Digit Recognition System(Real-Time)

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com/)

An interactive web application for recognizing handwritten digits using a CNN model trained on the MNIST dataset. 
Users can draw directly on a browser canvas, and the model will predict the digit in **real-time**.

## Features
- **CNN Model Training** on the MNIST dataset with a 90/10 train-validation split.
- **Advanced Data Augmentation**: Custom stroke-width adjustments (dilation/erosion) and geometric transformations to improve accuracy on real-world canvas drawings.
- **Flask Web Application** with `/predict` endpoint for inference.
- **Interactive Canvas** in the browser for drawing digits.
- **Real-Time Prediction** while drawing.
- **Preprocessing**: Cropping, scaling, and padding to make input resemble the MNIST format.

## Model Training & Architecture

### Data Augmentation Pipeline
To make the model robust to different handwriting styles on the web canvas, the training data passes through an augmentation pipeline:
*   **Geometric Transformations**: Random Rotation (±10°), Translation (±10%), and Zoom (±10%).
*   **Dynamic Stroke Width**: Randomly applies morphological operations (dilation and erosion) so the model learns to recognize both very thin and very thick drawn strokes.

### Convolutional Neural Network (CNN)
1. **Input**: 28×28 Grayscale Image
2. **Conv2D** (32 filters, 3×3 kernel, ReLU) → **MaxPooling2D**
3. **Conv2D** (64 filters, 3×3 kernel, ReLU) → **MaxPooling2D**
4. **Flatten** → **Dense** (128 units, ReLU)
5. **Dropout** (0.3 rate to prevent overfitting)
6. **Output**: Dense (10 units, Softmax) for classifying digits 0–9.

**Training Details:**
*   **Optimizer:** Adam
*   **Loss:** Sparse Categorical Crossentropy
*   **Callbacks:** Early Stopping (monitors validation accuracy, patience of 3 epochs, restores best weights).
*   **Output:** Saved in the modern Keras v3 format (`models/mnist_cnn.keras`).

