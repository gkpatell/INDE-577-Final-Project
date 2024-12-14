# PCA Analysis on a Synthetic Dataset

## Overview
This project demonstrates the application of **Principal Component Analysis (PCA)** to a synthetic dataset generated using `make_classification`. The goal is to reduce dimensionality, visualize the dataset effectively, and understand the significance of the principal components.

## Features
1. Generate a synthetic dataset with customizable features and labels.
2. Visualize pairwise relationships in the dataset using Seaborn.
3. Perform PCA to determine the optimal number of components based on the explained variance ratio.
4. Visualize the results in 2D and 3D spaces for better interpretation.

## Dataset
The dataset is generated using the `make_classification` function from the `sklearn.datasets` module. It contains:
- **1000 samples**
- **7 features** (4 informative and 3 redundant)

The dataset is well-suited for demonstrating PCA due to its multidimensional nature.

## Workflow
### 1. Data Generation
- A synthetic dataset is created with a mix of informative and redundant features.
- The feature matrix (`X`) and target labels (`y`) are derived.

### 2. Data Visualization
- Pairwise relationships in the dataset are visualized using Seaborn's `pairplot`.

### 3. Principal Component Analysis (PCA)
- PCA is applied to the dataset to analyze the explained variance ratio for each principal component.
- The cumulative explained variance ratio is plotted to identify the optimal number of components.

### 4. Dimensionality Reduction
- The dataset is transformed to the first three principal components for further analysis.

### 5. Visualization of Reduced Data
- A 2D scatter plot visualizes the first two principal components with color coding for target labels.
- A 3D scatter plot created using Plotly provides a detailed visualization of the first three components.

## Installation
Clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
