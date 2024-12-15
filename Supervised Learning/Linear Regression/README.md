# Linear Regression Model for Predicting California Housing Prices

This project demonstrates the development of a **Linear Regression Model** to predict **median house values** using the **California Housing Dataset**. It includes data preprocessing, model training, evaluation, exploratory data analysis (EDA), and saving the model for future use.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Requirements](#requirements)  
3. [Dataset](#dataset)  
4. [Steps](#steps)  
5. [Results](#results)  
6. [Usage](#usage)  

---

## Project Overview

This project builds a linear regression model to predict housing prices in California based on features like population density, median income, and housing occupancy. The workflow includes:  

- Data loading and preprocessing.  
- Splitting data into training and testing sets.  
- Building a model pipeline with preprocessing steps and the linear regression algorithm.  
- Model evaluation using metrics like **Mean Squared Error (MSE)** and **R-squared (R²)**.  
- Exploratory Data Analysis (EDA) to understand the data.  
- Residual and error analysis to diagnose the model's performance.  
- Saving the trained model for deployment.  

---

## Requirements

To run this project, ensure you have the following libraries installed:

- **numpy**  
- **pandas**  
- **matplotlib**  
- **seaborn**  
- **scikit-learn**  
- **joblib**

You can install the required packages using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

---

## Dataset

The project uses the **California Housing Dataset**, a popular dataset included in Scikit-learn.  
It contains information about:  

- **Median Income**  
- **Average House Age**  
- **Average Rooms**  
- **Population**  
- **Latitude and Longitude**  

The target variable is **Median House Value (MedHouseVal)**.

---

## Steps

### 1. **Import Required Libraries**
   - Load necessary Python libraries for data processing, modeling, and visualization.

### 2. **Load the Dataset**
   - Fetch the California housing dataset using Scikit-learn's `fetch_california_housing`.

### 3. **Data Preprocessing**
   - Check for missing values and handle them.
   - Separate features (independent variables) and the target (dependent variable).
   - Scale numerical features using **StandardScaler**.

### 4. **Split Data into Training and Testing Sets**
   - Use `train_test_split` to divide the dataset into 80% training and 20% testing data.

### 5. **Model Pipeline**
   - Build a pipeline with two steps:
     1. Scaling features using **StandardScaler**.
     2. Training a **Linear Regression** model.  

### 6. **Predictions and Evaluation**
   - Use the pipeline to predict median house values for the test set.  
   - Evaluate performance using:
     - **Mean Squared Error (MSE)**: Measures average squared difference between predicted and actual values.
     - **R-squared (R²)**: Indicates how well the model explains variance in the target variable.

### 7. **Exploratory Data Analysis (EDA)**
   - Visualize the distribution of the target variable.  
   - Analyze feature correlations using a **heatmap**.

### 8. **Error Analysis**
   - Plot residuals to check for model bias.  
   - Scatter plot actual vs. predicted values to visualize prediction accuracy.

### 9. **Save the Model**
   - Save the trained model pipeline using **joblib** for deployment.

---

## Results

- **Mean Squared Error (MSE):** Provides a quantitative measure of model prediction error.  
- **R-squared (R²):** Indicates the proportion of variance in the target variable explained by the model.  

### Sample Metrics (Example Output):
- MSE: **0.54**  
- R²: **0.72**

### Visualization:
1. **Residuals Distribution**: Analyzes the error spread and detects systematic errors.
2. **Predicted vs. Actual Plot**: Shows alignment between predicted and actual values.

---

## Usage

1. **Clone the repository** or download the script.  
2. Place your Python script and requirements in the same folder.  
3. Run the script:

   ```bash
   python linear_regression_model.py
   ```

4. **Load the saved model** for future predictions:

   ```python
   import joblib
   pipeline = joblib.load('linear_regression_pipeline.pkl')
   new_predictions = pipeline.predict(new_data)
   ```

---

## Future Work

1. Improve the model by incorporating polynomial regression or advanced techniques.  
2. Experiment with feature engineering for better results.  
3. Explore hyperparameter tuning for optimal performance.  
