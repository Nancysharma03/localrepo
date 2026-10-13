# AI-Assisted Pneumonia Detection from Chest X-ray Images Using CNN and ResNet50

A deep learning project for classifying chest X-ray images into **NORMAL** and **PNEUMONIA** categories using a Convolutional Neural Network (CNN) and ResNet50.

## Overview

This project explores AI-assisted pneumonia detection through medical image classification. It implements a baseline CNN and a ResNet50-based model to analyze chest X-ray images and compare their classification performance.

The project was developed using Python, TensorFlow/Keras, and Google Colab.

## Objectives

- Develop a CNN model for pneumonia classification.
- Explore ResNet50 for transfer learning and image classification.
- Preprocess and augment chest X-ray images.
- Evaluate the models using classification and diagnostic performance metrics.
- Explore the potential of deep learning for AI-assisted medical image analysis.

## Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)

**Source:** Kaggle

**Dataset link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains 5,856 pediatric chest X-ray images classified into two categories:

- **NORMAL:** Images without signs of pneumonia.
- **PNEUMONIA:** Images showing signs of pneumonia.

### Dataset Distribution

| Dataset split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,341 | 3,875 | 5,216 |
| Testing | 234 | 390 | 624 |
| Validation | 8 | 8 | 16 |
| **Total** | **1,583** | **4,273** | **5,856** |

**Note:** The notebook's training pipeline creates a 20% validation subset from the training directory using `ImageDataGenerator`. The original dataset also contains a separate validation directory with 16 images.

## Methodology

### 1. Data Preprocessing

- Resize images to 224 × 224 pixels.
- Normalize pixel values to the range 0–1.
- Apply data augmentation during training, including rotation, zoom, horizontal and vertical shifts, and horizontal flipping.
- Prepare training, validation, and testing image generators.

### 2. CNN Model

A custom Convolutional Neural Network is used as the baseline model for binary image classification. The architecture uses convolutional layers and other neural network components to learn image features.

### 3. ResNet50 Model

ResNet50 is explored as a deep learning architecture for chest X-ray classification using residual connections and transfer learning.

### 4. Model Evaluation

The models are evaluated using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Precision-recall analysis

Grad-CAM visualizations are also explored to help interpret image regions contributing to CNN predictions.

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV
- Google Colab
- Jupyter Notebook

## Trained Models

The project includes two trained model checkpoint files:

- `best_cnn_model.keras`
- `best_resnet50_model.keras`

These checkpoints are stored separately from the GitHub source repository because of their large file sizes.

## Repository Structure

```text
AI-Pneumonia-Detection/
├── Chapter_8_main.ipynb
├── README.md
└── .gitignore
```

Trained model files and the dataset are maintained separately and will be configured for the deployment application.

## How to Run

1. Open the notebook in Google Colab.
2. Download the dataset from the Kaggle source above.
3. Configure the dataset path in the notebook.
4. Install the required Python dependencies.
5. Execute the notebook cells in order to run preprocessing, model training, evaluation, and visualization.

To run inference using the existing trained checkpoints, load the appropriate `.keras` file with TensorFlow/Keras and apply the same preprocessing used during training.

## Future Work

- Develop an interactive web application using Streamlit.
- Enable chest X-ray image uploads and model predictions.
- Compare CNN and ResNet50 predictions through an interactive interface.
- Improve model interpretability and evaluate generalization on independent datasets.
- Deploy the application for educational demonstration.

## Disclaimer

This project is intended for educational and research purposes only. It is not a clinically validated diagnostic system and must not be used to diagnose patients or guide medical treatment.

## Author

**Nancy Sharma**

B.Tech. Computer Science and Engineering — Artificial Intelligence and Machine Learning

LNCT University, Bhopal
