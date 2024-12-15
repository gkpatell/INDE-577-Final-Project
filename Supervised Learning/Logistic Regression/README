# Logistic Regression Model: Titanic Survival Prediction

This project implements a **Logistic Regression Model** to predict **survival** on the Titanic using the **Titanic dataset**. The workflow includes data preprocessing, model building, evaluation, and visualization of results.

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

The Titanic dataset is used to predict whether a passenger survived the disaster based on features like age, fare, and class. This project uses Logistic Regression, a simple yet powerful algorithm for binary classification tasks. The key objectives are:  

1. **Data Preprocessing**: Cleaning and preparing data for modeling.  
2. **Model Building**: Training a Logistic Regression model.  
3. **Evaluation**: Assessing model performance using metrics like accuracy, ROC-AUC, and confusion matrix.  
4. **Error Analysis**: Understanding false positives and false negatives.  
5. **Visualization**: Analyzing the importance of features and model predictions.

---

## Requirements

Ensure the following libraries are installed:

- **numpy**  
- **pandas**  
- **matplotlib**  
- **seaborn**  
- **scikit-learn**

Install them using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Dataset

The **Titanic dataset** is sourced from Kaggle. It contains information about passengers, such as:

- **Survived**: Target variable (1 = survived, 0 = did not survive).  
- **Pclass**: Ticket class (1st, 2nd, 3rd).  
- **Sex**: Gender.  
- **Age**: Age of the passenger.  
- **SibSp**: Number of siblings/spouses aboard.  
- **Fare**: Ticket price.  

---

## Workflow

### Step 1: Import Required Libraries
Load the necessary libraries for data processing, visualization, and modeling.

### Step 2: Load and Explore the Dataset
- Download the dataset using the provided URL.  
- Analyze data structure and identify missing values.  
- Drop unnecessary columns (e.g., `Name`, `Ticket`, `Cabin`) to simplify the model.  

### Step 3: Data Preprocessing
- Handle missing values: Fill `Age` with the median and `Embarked` with the mode.  
- Encode categorical variables like `Sex` and `Embarked`.  
- Split the dataset into training and testing sets.  
- Standardize features using **StandardScaler**.

### Step 4: Train the Logistic Regression Model
- Train a **Logistic Regression** model using the training data.  

### Step 5: Evaluate the Model
- Compute metrics like:
  - **Accuracy**: Overall correctness of predictions.  
  - **Confusion Matrix**: Summarizes true/false positives and negatives.  
  - **Classification Report**: Includes precision, recall, and F1-score.

### Step 6: Performance and Error Analysis
- Analyze false positives and false negatives.  
- Investigate misclassified samples to identify patterns.  

### Step 7: Advanced Metrics and Visualization
- Calculate **ROC-AUC** to evaluate classifier performance.  
- Plot:
  - **ROC Curve**: Trade-off between true positive rate and false positive rate.  
  - **Precision-Recall Curve**: Performance in imbalanced datasets.  

### Step 8: Visualize Feature Importance
- Plot feature coefficients to understand their impact on predictions.  

---

## Results

### Metrics:
- **Accuracy**: ~80%  
- **ROC-AUC**: ~0.85  

### Visualizations:
1. **Confusion Matrix**: Highlights correct and incorrect predictions.  
2. **ROC and Precision-Recall Curves**: Demonstrate model reliability.  
3. **Feature Importance**: Reveals key predictors like `Sex_male`, `Pclass`, and `Fare`.

### Example Output:

- **False Positives**: Passengers misclassified as survivors.  
- **False Negatives**: Survivors misclassified as non-survivors.  

---

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/titanic-logistic-regression.git
   cd titanic-logistic-regression
   ```

2. **Run the Script**:
   ```bash
   python logistic_regression_titanic.py
   ```

3. **Analyze Outputs**:
   Review the metrics, plots, and misclassified samples.  

---

## Future Enhancements

1. **Feature Engineering**:
   - Extract features like family size and title from names.  

2. **Hyperparameter Tuning**:
   - Optimize `C` (regularization strength) and `penalty` type for better performance.  

3. **Advanced Models**:
   - Compare Logistic Regression with other classifiers like Random Forest or Gradient Boosting.  
