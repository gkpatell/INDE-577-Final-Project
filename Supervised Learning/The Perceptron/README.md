# Perceptron Model: Heart Disease Classification

This project demonstrates the use of a Perceptron model to classify whether a patient has heart disease based on various health features. The dataset used is from the UCI Machine Learning Repository, specifically the Cleveland heart disease dataset. Below is a breakdown of the key steps involved in the implementation.

## Project Steps Overview:

### Step 1: Import Required Libraries
We begin by importing the necessary libraries for data manipulation, visualization, machine learning model creation, and evaluation:
- `numpy`, `pandas` for data handling
- `matplotlib`, `seaborn` for visualization
- `sklearn` for machine learning model implementation and evaluation

### Step 2: Load and Explore the Dataset
The dataset is loaded directly from the UCI repository, which contains features such as age, sex, cholesterol levels, and ECG results. Missing values are handled by dropping rows with missing data.

### Step 3: Preprocessing the Data
- **Feature Selection**: The target variable is separated from the features.
- **Train-Test Split**: The data is split into training and testing sets using an 80-20 split, ensuring stratification based on the target variable.
- **Standardization**: The feature values are standardized to have a mean of 0 and a standard deviation of 1.
- **Polynomial Feature Generation**: Polynomial features are created to potentially capture higher-order relationships between features.

### Step 4: Implement the Perceptron Algorithm
- **Perceptron Model**: A Perceptron model is trained on both the original and polynomial feature sets.
- **Model Training**: The model is trained using the `Perceptron` class from `sklearn` with a maximum of 1000 iterations and a tolerance of 1e-3 for convergence.

### Step 5: Evaluate the Model
- **Accuracy**: The accuracy of the model on both the original and polynomial feature sets is computed.
- **Classification Report**: Precision, recall, and F1-scores are displayed in a classification report.
- **Confusion Matrix**: A confusion matrix is plotted to visualize the model's performance in terms of false positives and false negatives.
- **Model Calibration**: A calibrated version of the Perceptron model is created to predict probabilities and improve model performance.

### Step 6: Performance and Error Analysis
- **False Positives/Negatives**: An analysis of the number of false positives and false negatives is performed.
- **Misclassified Samples**: The incorrectly classified samples are displayed for further investigation.
- **ROC and AUC**: Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) scores are plotted to evaluate model performance across different thresholds.
- **Precision-Recall Curves**: Precision-recall curves are plotted to assess the model's precision and recall at various thresholds.

### Step 7: Visualizing Decision Boundaries
- **2D Decision Boundary Visualization**: The decision boundaries of the Perceptron model are visualized using the first two features. A contour plot is generated to display the regions classified as heart disease or not.

## Key Libraries and Tools:
- `Perceptron` from `sklearn.linear_model`: Implements the perceptron classification algorithm.
- `StandardScaler` from `sklearn.preprocessing`: Scales features to have zero mean and unit variance.
- `PolynomialFeatures` from `sklearn.preprocessing`: Generates polynomial and interaction features.
- `train_test_split` from `sklearn.model_selection`: Splits the data into training and testing sets.
- `classification_report`, `confusion_matrix` from `sklearn.metrics`: Used for evaluating the model's performance.

## How to Run:
1. Install required libraries if not already installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Run the Python script to execute the workflow and see results such as model performance, confusion matrix, and ROC curve.

## Results:
The output of the model includes:
- Accuracy, precision, recall, and F1-scores for both the basic Perceptron model and the model with polynomial features.
- A confusion matrix visualization to help understand the classification performance.
- ROC and precision-recall curve plots to assess the model's ability to distinguish between classes at various thresholds.
