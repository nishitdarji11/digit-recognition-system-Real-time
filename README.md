# MNIST Handwritten Digit Recognition - Flask + Realtime Canvas

# Digit Recognition System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com/)

An interactive web application for recognizing handwritten digits using a CNN model trained on the MNIST dataset.  
Users can draw directly on a browser canvas, and the model will predict the digit in **realtime**.



## Features
- **CNN model training** on the MNIST dataset.
- **Flask web application** with `/predict` endpoint for inference.
- **Interactive canvas** in the browser for drawing digits.
- **Realtime prediction** while drawing.
- **Preprocessing**: cropping, scaling, and padding to make input resemble MNIST format.


## Model Architecture

Simple CNN model:

1. Conv2D(32, kernel 3×3, ReLU) → MaxPooling2D
2. Conv2D(64, kernel 3×3, ReLU) → MaxPooling2D
3. Flatten → Dense(128, ReLU) → Dropout(0.3)
4. Dense(10, Softmax) for classifying digits 0–9.

Optimizer: **Adam**  
Loss: **SparseCategoricalCrossentropy**  
Accuracy on MNIST: ±98%.

## License


