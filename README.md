# Food Image Classifier  

*A convolutional neural network (CNN) project to classify food images from the Food-101 dataset.*

---

## Table of Contents  
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Architecture](#model-architecture)  
- [Future Work](#future-work)   

---

## Overview  
This project implements a deep learning-based food image classifier using the Food-101 dataset. The goal is to train a convolutional neural network (CNN) that can recognize and classify images into 101 different food categories with high accuracy. It demonstrates data preprocessing, model training, and evaluation in Python using TensorFlow and Keras.  

---

## Dataset  
The Food-101 dataset contains 101,000 images of 101 food categories, with 1,000 images per category. The dataset is publicly available and can be downloaded from [Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101) .

- **Classes**: 101 food categories  
- **Train/Validation Split**: 80% Training, 20% Validation  
- **Image Size**: Resized to 150x150 for model input  

---

## Installation  
Clone the repository:  
   ```bash
   git clone https://github.com/ahmdmohamedd/food-image-classifier.git
   cd food-image-classifier
   ```

---

## Usage  
1. **Run the Jupyter Notebook**  
   Launch the project notebook:  
   ```bash
   jupyter notebook food_image_classifier.ipynb
   ```

2. **Train the Model**  
   The notebook includes all steps for loading the dataset, data augmentation, model training, and evaluation. Modify parameters like `epochs` and `batch size` as needed.

3. **Test the Model**  
   The final section in the notebook demonstrates how to test the model on new images.

---

## Model Architecture  
The convolutional neural network (CNN) model consists of:  
- **Input Layer**: 150x150 RGB images  
- **Conv Layers**: 3 convolutional blocks with ReLU activation and max pooling  
- **Dense Layers**: Fully connected layers for feature extraction  
- **Dropout Layer**: For regularization to prevent overfitting  
- **Output Layer**: Softmax activation for classification into 101 categories  

---

## Future Work  
- Experiment with deeper architectures like ResNet or EfficientNet.  
- Implement transfer learning to improve accuracy and reduce training time.  
- Extend the model to recognize multi-class food combinations.  

---
