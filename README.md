# Breast Cancer Image Classification

This project focuses on the classification of breast cancer images using machine learning techniques. The dataset used for this project is sourced from Kaggle and contains breast histopathology images.

## Dataset
The dataset can be found at the following link: [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)

## Project Structure
- `notebooks/`: Jupyter notebooks containing data exploration, preprocessing, and model training.
- `frontend/`: React application for visualizing and interacting with the model predictions.
- `models/`: Saved machine learning models.
- `data/`: Directory for storing raw and processed data.

## Models
The project utilizes deep learning models for breast cancer image classification. The following libraries and methods are used:

### Libraries Used
- **TensorFlow**: An open-source machine learning library used for training and deploying models.
- **Keras**: A high-level neural networks API, written in Python and capable of running on top of TensorFlow.
- **NumPy**: A fundamental package for scientific computing with Python, used for numerical operations.
- **Pillow**: A Python Imaging Library (PIL) fork used for opening, manipulating, and saving image files.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.

### Methodology
#### Data Preprocessing:
- Images are resized and normalized to ensure consistency.
- Data augmentation techniques such as rotation, flipping, and zooming are applied to increase the variability of the training set.

#### Model Architecture:
- A Convolutional Neural Network (CNN) is designed to extract features from the images.
- The architecture typically includes multiple convolutional layers followed by pooling layers, fully connected layers, and an output layer with softmax activation for classification.

#### Training:
- The model is trained using the training dataset with a specified number of epochs and batch size.
- The Adam optimizer is commonly used for optimization, and categorical cross-entropy is used as the loss function.

#### Evaluation:
- The trained model is evaluated on a separate validation set to assess its performance.
- Metrics such as accuracy, precision, recall, and F1-score are calculated to evaluate the model's effectiveness.

#### Model Saving:
- The best-performing model is saved for future inference and deployment.

## Setup and Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SiddhiDabholkar10/Breast-Cancer-Image-Classification.git
   cd Breast-Cancer-Image-Classification
