# Neural Network Models for MNIST and CIFAR-10 Classification

This project demonstrates the application of neural networks for image classification tasks using the **MNIST** (handwritten digits) and **CIFAR-10** (color images) datasets. The implementation covers model training, evaluation, and saving/loading models for future predictions.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Requirements](#requirements)  
3. [Datasets](#datasets)  
4. [Workflow](#workflow)  
5. [Results](#results)  
6. [Usage](#usage)  
7. [Future Enhancements](#future-enhancements)  

---

## Project Overview

This project utilizes TensorFlow's **Keras API** to build, train, and evaluate deep learning models. Key objectives include:

1. **MNIST Dataset**:
   - Build a dense neural network for grayscale 28x28 images of digits (0-9).  
   - Train the model and visualize predictions.  

2. **CIFAR-10 Dataset**:
   - Implement a deeper neural network with dropout and batch normalization for 32x32 color images.  
   - Analyze model performance using metrics and visualizations like accuracy curves and confusion matrices.

---

## Requirements

Ensure the following libraries are installed:

- **tensorflow**  
- **numpy**  
- **matplotlib**  
- **scikit-learn**

Install them using:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## Datasets

### MNIST Dataset:
- Handwritten digits (0-9), grayscale, size: **28x28 pixels**.
- Classes: 10.  
- Source: [Keras MNIST dataset](https://keras.io/api/datasets/mnist/).

### CIFAR-10 Dataset:
- 10 categories of 32x32 color images, including airplanes, cars, and animals.  
- Classes: 10.  
- Source: [Keras CIFAR-10 dataset](https://keras.io/api/datasets/cifar10/).

---

## Workflow

### Step 1: Import Required Libraries
Load TensorFlow and other libraries for preprocessing, modeling, and visualization.

### Step 2: Load and Preprocess Datasets
- **Normalize Images**: Rescale pixel values to the range [0, 1].  
- **One-hot Encoding**: Not required, as we use `sparse_categorical_crossentropy`.  

### Step 3: MNIST Neural Network
1. **Architecture**:
   - Input: Flattened 28x28 image (784 features).  
   - Two hidden layers with 128 neurons and ReLU activation.  
   - Output: 10 neurons with Softmax for multi-class classification.  

2. **Training**:
   - Optimizer: Adam.  
   - Loss: Sparse categorical crossentropy.  
   - Metrics: Accuracy.  

3. **Evaluation**:
   - Calculate test loss and accuracy.  
   - Visualize predictions and sample images.  

4. **Save and Load Model**:
   - Save the trained model as `epic_num_reader.h5`.  
   - Load the saved model for inference.

---

### Step 4: CIFAR-10 Neural Network
1. **Architecture**:
   - Flatten input (32x32x3 = 3072 features).  
   - Dense layers with 512 and 256 neurons, ReLU activation.  
   - Dropout layers (50%) to prevent overfitting.  
   - Batch normalization for stable training.  
   - Output: 10 neurons with Softmax for classification.

2. **Training**:
   - Epochs: 10 with validation on test data.  
   - Track performance using accuracy and loss curves.

3. **Evaluation**:
   - Plot training and validation accuracy/loss.  
   - Display a confusion matrix and a classification report.  

4. **Save and Load Model**:
   - Save the trained model as `cifar10_model.h5`.  
   - Reload for predictions on test images.

---

## Results

### MNIST:
- **Accuracy**: ~98% on test data.  
- **Loss**: Minimal training and validation loss.  
- **Prediction Example**:
  - Correctly predicted the digit `5` from a sample test image.

### CIFAR-10:
- **Accuracy**: ~65% on test data (basic architecture).  
- **Loss**: Training and validation loss converge over 10 epochs.  
- **Confusion Matrix**: Highlights errors between similar categories (e.g., cats and dogs).  
- **Prediction Example**:
  - Correctly predicted `airplane` from a sample test image.

---

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/nn-mnist-cifar10.git
   cd nn-mnist-cifar10
   ```

2. **Run the Script for MNIST**:
   ```bash
   python mnist_nn.py
   ```

3. **Run the Script for CIFAR-10**:
   ```bash
   python cifar10_nn.py
   ```

4. **Saved Models**:
   - `epic_num_reader.h5`: Trained MNIST model.  
   - `cifar10_model.h5`: Trained CIFAR-10 model.

5. **Predictions**:
   - Replace `x_test[0]` with any test sample for inference.  

---

## Future Enhancements

1. **Improved Architectures**:
   - Integrate convolutional layers (CNNs) for better image feature extraction.  
   - Add global average pooling for efficient feature reduction.

2. **Hyperparameter Tuning**:
   - Optimize learning rates, layer sizes, and dropout rates.

3. **Augmentation**:
   - Apply data augmentation to CIFAR-10 to boost accuracy.

4. **Transfer Learning**:
   - Use pre-trained models like VGG16 or ResNet for CIFAR-10.
