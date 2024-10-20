# Multi Class Image Classification Using Tranfer Learning
# Introduction
Dog breed classification is a challenging task due to the high variability of appearances within breeds and similarities among them. This project addresses these challenges by leveraging InceptionResNetV2, a pre-trained model, in conjunction with Transfer Learning techniques to enhance accuracy and reduce computational costs.
# Project Structure
.
├── dataset/
│   ├── train/
│   ├── test/
│   ├── labels.csv
│   └── sample_submission.csv
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── model.py
│   └── preprocess.py
├── results/
│   ├── model.h5
│   └── evaluation_metrics.txt
└── README.md
# Dataset
Source: Dog Breed Identification Dataset
Classes: 120 dog breeds
Total Images: 20,580 images
Train/Test Split: 80/20
The dataset consists of images in various lighting conditions and backgrounds, making it a great fit for evaluating the robustness of machine learning models.

# Data Preprocessing
We apply several preprocessing techniques:

JPEG Conversion: Standardizes images for input to the model.
Background Removal: Focuses attention on the dog in each image.
Data Augmentation: Includes random flips, rotations, and scaling to improve generalization.

# Model Architecture
The primary model used is InceptionResNetV2:

Pre-trained on ImageNet, with fine-tuning applied to adapt the model for 120 dog breeds.
Key Layers:
Batch Normalization
GlobalAveragePooling2D
Dense Layers with ReLU activation
Softmax output layer for 120 breed classes
# Installation
Prerequisites
Python 3.8+
TensorFlow 2.x or Keras
NumPy, Pandas, OpenCV, and Matplotlib
# Architecture
# Results
Train Accuracy : 93.75%
Validation Accuracy : 96.42%
Test Accuracy : 89.6771%
# Predictions

