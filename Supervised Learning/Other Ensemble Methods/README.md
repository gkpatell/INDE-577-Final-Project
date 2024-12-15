# Customer Churn Prediction with Ensemble Methods

This project demonstrates the use of **Ensemble Methods** for predicting customer churn. Ensemble techniques like Random Forest, Bagging, and Gradient Boosting are utilized to analyze their effectiveness in classification tasks.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Requirements](#requirements)  
3. [Dataset](#dataset)  
4. [Workflow](#workflow)  
5. [Results](#results)  
6. [Usage](#usage)  
7. [Future Enhancements](#future-enhancements)

---

## Project Overview

The primary goal of this project is to predict whether a customer will churn based on demographic and account-related features. It includes:
- **Random Forest Classifier**
- **Bagging Classifier**
- **Gradient Boosting Classifier**

The models are evaluated for accuracy and feature importance, with insights provided for feature contributions.

---

## Requirements

To run this project, install the following libraries:

- **pandas**  
- **numpy**  
- **matplotlib**  
- **seaborn**  
- **scikit-learn**  
- **joblib**  

Install using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## Dataset

### Customer Churn Data
- **Source**: Public dataset available at [GitHub repository](https://github.com/srees1988/predict-churn-py).  
- **Description**:  
  Contains customer demographic, account, and service details with the target column `Churn`, indicating if a customer left the service (`Yes`/`No`).  

Key features:
- **Numerical**: `TotalCharges`, `MonthlyCharges`, `SeniorCitizen`.  
- **Categorical**: `gender`.  

---

## Workflow

### Step 1: Import Libraries and Load Data
Load the dataset using `pandas` and explore its structure.

### Step 2: Data Cleaning and Preprocessing
1. Handle missing values in the `TotalCharges` column.  
2. Encode categorical variables like `gender` using one-hot encoding.  
3. Map the target column (`Churn`) to binary values (1 for `Yes`, 0 for `No`).  
4. Select features: `SeniorCitizen`, `MonthlyCharges`, `TotalCharges`, and `gender_Male`.  

### Step 3: Train-Test Split
Split the data into training (80%) and testing (20%) sets.

### Step 4: Implement Ensemble Methods
Train and evaluate three ensemble classifiers:
1. **Random Forest Classifier**: A robust method combining bagging and feature randomness for better predictions.  
2. **Bagging Classifier**: Uses an ensemble of Random Forest models for added stability.  
3. **Gradient Boosting Classifier**: Focuses on minimizing classification errors using sequential training.

---

## Results

### Model Evaluation
Each model's performance is measured using **classification report** and **accuracy score**.

- **Random Forest**: 
  - Accuracy: **X%**  
  - Strengths: Handles non-linear relationships and highlights feature importance.

- **Bagging Classifier**: 
  - Accuracy: **X%**  
  - Strengths: Reduces overfitting and improves stability.

- **Gradient Boosting**: 
  - Accuracy: **X%**  
  - Strengths: Effectively minimizes classification errors with iterative learning.

### Feature Importance
Analyzed the importance of input features using the Random Forest model. Visualized with a bar chart:
- Features like `TotalCharges` and `MonthlyCharges` showed significant importance in predicting churn.

---

## Usage

### Steps to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/customer-churn-ensemble.git
   cd customer-churn-ensemble
   ```

2. **Run the Script**:
   ```bash
   python churn_prediction.py
   ```

3. **Model and Output**:
   - Model: Saved as `random_forest_model.pkl` using `joblib`.  
   - Predictions: Stored and evaluated in the script.  

---

## Visualization

### Feature Importance:
- Bar chart highlighting the contribution of features like `TotalCharges` and `MonthlyCharges`.  

### Confusion Matrix:
- Visualize model performance in distinguishing churned vs. retained customers.

---

## Future Enhancements

1. **Hyperparameter Tuning**:
   - Utilize `GridSearchCV` or `RandomizedSearchCV` for optimizing model parameters.

2. **Additional Features**:
   - Include new features like customer tenure or service history for better predictions.

3. **Advanced Ensemble Models**:
   - Explore **XGBoost** or **LightGBM** for improved performance.

4. **Cross-Validation**:
   - Implement cross-validation for better generalization.
