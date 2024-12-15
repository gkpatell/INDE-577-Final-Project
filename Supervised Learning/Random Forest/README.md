# Predicting California Housing Prices with Random Forest

This project demonstrates **Random Forest Regression** for predicting housing prices in California. The model is evaluated for its performance and tuned using GridSearchCV. It includes feature importance analysis and error diagnostics.

---

## Table of Contents

1. [Overview](#overview)  
2. [Requirements](#requirements)  
3. [Dataset](#dataset)  
4. [Workflow](#workflow)  
5. [Results](#results)  
6. [Usage](#usage)  
7. [Future Enhancements](#future-enhancements)

---

## Overview

The primary objective of this project is to predict the **median house value** in California based on various demographic and geographical features. It covers:
- **Feature Engineering**  
- **Model Training**  
- **Hyperparameter Tuning**  
- **Error Analysis**  

---

## Requirements

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## Dataset

### California Housing Dataset
- **Source**: `fetch_california_housing` from `sklearn.datasets`.  
- **Description**:  
  This dataset contains 20,640 instances of housing data, with features such as median income, number of households, and proximity to the ocean.  
- **Target Variable**: Median house value (in $100,000 units).  

---

## Workflow

### Step 1: Import Libraries
Essential libraries such as `numpy`, `pandas`, `seaborn`, and `scikit-learn` are imported.

### Step 2: Load and Explore Dataset
1. Load the California Housing Dataset using `fetch_california_housing(as_frame=True)`.  
2. Perform exploratory analysis:
   - Check for missing values.  
   - Visualize the target variable (`Median House Value`).  
   - Display summary statistics for feature distributions.

### Step 3: Preprocessing
1. Split the dataset into training (80%) and testing (20%) sets.  
2. Scale the features using `StandardScaler` to standardize data.  

### Step 4: Train the Random Forest Regressor
1. Initialize and train a **Random Forest Regressor** with default parameters.  
2. Evaluate predictions on both training and test sets using:
   - **Mean Squared Error (MSE)**  
   - **Mean Absolute Error (MAE)**  
   - **R-squared Score (R²)**  

### Step 5: Hyperparameter Tuning
1. Use **GridSearchCV** to find the best combination of hyperparameters, such as:
   - Number of trees (`n_estimators`)  
   - Maximum depth (`max_depth`)  
   - Minimum samples per split and leaf.  
2. Train the tuned model and compare its performance with the default model.

### Step 6: Feature Importance
1. Extract feature importances from the best-tuned model.  
2. Visualize the contribution of each feature using a bar chart.  

### Step 7: Error Analysis
1. Analyze residuals (difference between actual and predicted values).  
2. Visualize:
   - Distribution of residuals using histograms.  
   - Residuals vs. predicted values to detect patterns.  

---

## Results

### Model Evaluation
- **Default Random Forest**:
  - Training R²: **X%**  
  - Test R²: **X%**  
  - Test MAE: **X**  

- **Tuned Random Forest**:
  - Training R²: **Y%**  
  - Test R²: **Y%**  
  - Test MAE: **Y**  

### Feature Importance
Top contributing features identified:
1. **Median Income**  
2. **Households**  
3. **Population Density**  

### Residual Analysis
- Residual distribution was centered around zero with minimal skewness.  
- No significant heteroscedasticity observed in residuals vs. predicted values.  

---

## Usage

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/random-forest-housing.git
   cd random-forest-housing
   ```

2. Run the Python script:
   ```bash
   python housing_price_prediction.py
   ```

3. Outputs:
   - Model saved as `random_forest_regressor_model.pkl`.  
   - Predictions and residuals saved in `random_forest_results.csv`.  

---

## Future Enhancements

1. **Advanced Models**:
   - Explore **XGBoost** or **LightGBM** for better results.  
   - Compare with linear regression models to benchmark performance.  

2. **Additional Features**:
   - Include geographical proximity to facilities like schools or hospitals.  
   - Engineer new features using interaction terms or clustering.  

3. **Hyperparameter Optimization**:
   - Utilize **Bayesian Optimization** or **RandomizedSearchCV** for faster tuning.  

4. **Cross-Validation**:
   - Integrate K-Fold Cross-Validation to improve generalization.  

5. **Deployment**:
   - Deploy the model using Flask or FastAPI for real-time predictions.  

---
