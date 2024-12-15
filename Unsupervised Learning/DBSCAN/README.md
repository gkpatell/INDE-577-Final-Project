# DBSCAN Clustering Model

This repository demonstrates the implementation of the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm for clustering customer data based on their annual income and spending score. DBSCAN is a density-based clustering algorithm that groups points that are closely packed together, while marking outliers as noise. Unlike K-Means, DBSCAN can find arbitrarily shaped clusters and does not require the number of clusters to be specified beforehand.

## Key Steps in the Model:

### 1. **Data Loading and Exploration**
The dataset used in this implementation is the "Mall_Customers.csv.xls", which contains customer data including features such as annual income and spending score.

```python
df = pd.read_csv('Mall_Customers.csv.xls')
```

The data is explored by viewing the first few rows and the general shape of the dataset.

### 2. **Data Preparation**
We extract the relevant features (Annual Income and Spending Score) from the dataset and store them in the variable `X`.

```python
X = df.iloc[:, [3, 4]].values
```

### 3. **K-Means Elbow Method (For Comparison)**
To determine the optimal number of clusters, we apply the K-Means algorithm and use the **Elbow Method** to visualize the within-cluster sum of squares (WCSS) for various numbers of clusters (1 to 10).

```python
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

This helps in identifying an appropriate number of clusters based on the sharp decrease in WCSS.

### 4. **Applying DBSCAN Clustering**
We apply the DBSCAN algorithm to the dataset to discover clusters based on the density of data points.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=5, min_samples=5)
labels = dbscan.fit_predict(X)
```

### 5. **Visualizing the Clusters**
The dataset is visualized, showing the identified clusters using different colors. Points marked with `-1` are considered noise.

```python
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], s=10, c='black')
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], s=10, c='blue')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], s=10, c='red')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], s=10, c='green')
plt.scatter(X[labels == 3, 0], X[labels == 3, 1], s=10, c='brown')
plt.scatter(X[labels == 4, 0], X[labels == 4, 1], s=10, c='pink')
plt.scatter(X[labels == 5, 0], X[labels == 5, 1], s=10, c='yellow')
plt.scatter(X[labels == 6, 0], X[labels == 6, 1], s=10, c='orange')

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Clusters of Customers')
plt.show()
```

### 6. **Cluster Visualization**
The resulting plot shows various clusters of customers based on their annual income and spending score, with different colors representing different clusters. Black points indicate noise (outliers).

## Key Parameters of DBSCAN:
- **eps (epsilon)**: Defines the maximum distance between two points for them to be considered as neighbors.
- **min_samples**: The minimum number of points required to form a dense region (cluster).

These parameters control the density of the clusters and can be adjusted depending on the dataset.

## Requirements

To run this DBSCAN implementation, you need to have the following libraries installed:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Conclusion

This model demonstrates the power of DBSCAN for clustering data with varying densities and identifying outliers without requiring the number of clusters to be predefined. The algorithm works well for real-world datasets with noise and varying cluster shapes, making it a useful tool for unsupervised learning applications.
