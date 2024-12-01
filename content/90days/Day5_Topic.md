<div style="text-align: center;">
  <h1>🚀 Becoming a Scikit-Learn Boss in 90 Days: Day 5 – Unsupervised Learning: Clustering and Dimensionality Reduction 🧩📉</h1>
  <p>Unlock the Power of Unsupervised Techniques to Discover Hidden Patterns in Your Data!</p>
</div>

---

## 📑 Table of Contents

1. [🌟 Welcome to Day 5](#welcome-to-day-5)
2. [🔍 Review of Day 4 📜](#review-of-day-4-📜)
3. [🧠 Introduction to Unsupervised Learning 🧠](#introduction-to-unsupervised-learning-🧠)
    - [📚 What is Unsupervised Learning?](#what-is-unsupervised-learning-📚)
    - [🔍 Types of Unsupervised Learning Problems](#types-of-unsupervised-learning-problems-🔍)
4. [📊 Clustering Algorithms 📊](#clustering-algorithms-📊)
    - [🔵 K-Means Clustering](#k-means-clustering-🔵)
    - [🌳 Hierarchical Clustering](#hierarchical-clustering-🌳)
    - [🌀 DBSCAN](#dbscan-🌀)
5. [📉 Dimensionality Reduction Techniques 📉](#dimensionality-reduction-techniques-📉)
    - [📈 Principal Component Analysis (PCA)](#principal-component-analysis-pca-📈)
    - [🕵️‍♂️ t-Distributed Stochastic Neighbor Embedding (t-SNE)](#t-distributed-stochastic-neighbor-embedding-tsne-🕵️‍♂️)
6. [🛠️ Implementing Clustering and Dimensionality Reduction with Scikit-Learn 🛠️](#implementing-clustering-and-dimensionality-reduction-with-scikit-learn-🛠️)
    - [🔵 K-Means Example 🔵](#k-means-example-🔵)
    - [🌳 Hierarchical Clustering Example 🌳](#hierarchical-clustering-example-🌳)
    - [🌀 DBSCAN Example 🌀](#dbscan-example-🌀)
    - [📈 PCA Example 📈](#pca-example-📈)
    - [🕵️‍♂️ t-SNE Example 🕵️‍♂️](#tsne-example-🕵️‍♂️)
7. [📈 Model Evaluation for Unsupervised Learning 📈](#model-evaluation-for-unsupervised-learning-📈)
    - [🔍 Silhouette Score](#silhouette-score-🔍)
    - [📏 Davies-Bouldin Index](#davies-bouldin-index-📏)
    - [📐 Elbow Method](#elbow-method-📐)
8. [🛠️📈 Example Project: Customer Segmentation 🛠️📈](#example-project-customer-segmentation-🛠️📈)
    - [📋 Project Overview](#project-overview-📋)
    - [📝 Step-by-Step Guide](#step-by-step-guide-📝)
        - [1. Load and Explore the Dataset](#1-load-and-explore-the-dataset)
        - [2. Data Preprocessing](#2-data-preprocessing)
        - [3. Clustering with K-Means](#3-clustering-with-k-means)
        - [4. Clustering with DBSCAN](#4-clustering-with-dbscan)
        - [5. Dimensionality Reduction with PCA](#5-dimensionality-reduction-with-pca)
        - [6. Visualization with t-SNE](#6-visualization-with-tsne)
        - [7. Evaluating Clustering Performance](#7-evaluating-clustering-performance)
    - [📊 Results and Insights](#results-and-insights-📊)
9. [🚀🎓 Conclusion and Next Steps 🚀🎓](#conclusion-and-next-steps-🚀🎓)
10. [📜 Summary of Day 5 📜](#summary-of-day-5-📜)

---

## 1. 🌟 Welcome to Day 5

Welcome to **Day 5** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, we'll dive into **Unsupervised Learning**, focusing on **Clustering** and **Dimensionality Reduction** techniques. These methods are essential for discovering hidden patterns and reducing the complexity of your data, enabling more insightful analyses and efficient modeling.

---

## 2. 🔍 Review of Day 4 📜

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Model Evaluation and Selection**: Learned about cross-validation, hyperparameter tuning, and strategies to select the best model.
- **Bias-Variance Tradeoff**: Understood the balance between bias and variance to improve model generalization.
- **Model Validation Techniques**: Explored Train-Test Split, K-Fold Cross-Validation, Stratified K-Fold, and Leave-One-Out Cross-Validation.
- **Hyperparameter Tuning**: Mastered Grid Search, Randomized Search, and Bayesian Optimization for tuning model parameters.
- **Comparing Models**: Compared different regression models using performance metrics and visualizations.
- **Example Project**: Developed a regression pipeline to predict housing prices, evaluated multiple models, and optimized their performance through cross-validation and hyperparameter tuning.

With this foundation, we're ready to explore unsupervised techniques that will help you uncover hidden structures in your data.

---

## 3. 🧠 Introduction to Unsupervised Learning 🧠

### 📚 What is Unsupervised Learning? 📚

**Unsupervised Learning** is a type of machine learning where the model is trained on data without explicit labels. The goal is to identify underlying patterns, groupings, or structures within the data. Unlike supervised learning, which predicts outcomes based on labeled data, unsupervised learning discovers the inherent structure of the input data.

### 🔍 Types of Unsupervised Learning Problems 🔍

- **Clustering**: Grouping similar data points together based on feature similarities.
- **Dimensionality Reduction**: Reducing the number of features in a dataset while preserving important information.
- **Anomaly Detection**: Identifying unusual data points that do not conform to the expected pattern.
- **Association Rule Learning**: Discovering interesting relations between variables in large databases.

---

## 4. 📊 Clustering Algorithms 📊

Clustering algorithms aim to partition data into distinct groups where data points in the same group are more similar to each other than to those in other groups.

### 🔵 K-Means Clustering 🔵

A popular partitioning method that divides data into K clusters by minimizing the variance within each cluster.

**Key Features:**

- Simple and efficient for large datasets.
- Assumes spherical cluster shapes.
- Requires specifying the number of clusters (K) in advance.

### 🌳 Hierarchical Clustering 🌳

Builds a hierarchy of clusters either through agglomerative (bottom-up) or divisive (top-down) approaches.

**Key Features:**

- Does not require specifying the number of clusters beforehand.
- Can capture nested clusters.
- Computationally intensive for large datasets.

### 🌀 DBSCAN 🌀

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) groups together points that are closely packed and marks points in low-density regions as outliers.

**Key Features:**

- Identifies clusters of arbitrary shapes.
- Does not require specifying the number of clusters.
- Handles noise effectively.

---

## 5. 📉 Dimensionality Reduction Techniques 📉

Dimensionality reduction reduces the number of input variables in a dataset, enhancing computational efficiency and mitigating the curse of dimensionality.

### 📈 Principal Component Analysis (PCA) 📈

A linear technique that transforms data into a set of orthogonal components, capturing the maximum variance in the data.

**Key Features:**

- Reduces dimensionality while preserving variance.
- Helps in visualizing high-dimensional data.
- Assumes linear relationships between features.

### 🕵️‍♂️ t-Distributed Stochastic Neighbor Embedding (t-SNE) 🕵️‍♂️

A non-linear technique primarily used for data visualization by reducing data to two or three dimensions.

**Key Features:**

- Captures complex relationships and cluster structures.
- Computationally intensive for large datasets.
- Primarily used for visualization, not feature reduction for modeling.

---

## 6. 🛠️ Implementing Clustering and Dimensionality Reduction with Scikit-Learn 🛠️

### 🔵 K-Means Example 🔵

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming X is your feature set
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Visualize the clusters
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels, palette='viridis')
plt.title('K-Means Clustering')
plt.show()
```

### 🌳 Hierarchical Clustering Example 🌳

```python
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the model
hierarchical = AgglomerativeClustering(n_clusters=3)
labels = hierarchical.fit_predict(X)

# Visualize the clusters
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels, palette='magma')
plt.title('Hierarchical Clustering')
plt.show()
```

### 🌀 DBSCAN Example 🌀

```python
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the model
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualize the clusters
sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=labels, palette='coolwarm')
plt.title('DBSCAN Clustering')
plt.show()
```

### 📈 PCA Example 📈

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize PCA to reduce to 2 components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels  # Assuming clustering labels are available

# Visualize the PCA
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='Set2')
plt.title('PCA of Dataset')
plt.show()
```

### 🕵️‍♂️ t-SNE Example 🕵️‍♂️

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(X)

# Create a DataFrame for visualization
tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
tsne_df['Cluster'] = labels  # Assuming clustering labels are available

# Visualize t-SNE
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster', data=tsne_df, palette='deep')
plt.title('t-SNE of Dataset')
plt.show()
```

---

## 7. 📈 Model Evaluation for Unsupervised Learning 📈

Evaluating unsupervised models can be challenging since there are no ground truth labels. However, several metrics help assess the quality of clustering and dimensionality reduction.

### 🔍 Silhouette Score 🔍

Measures how similar an object is to its own cluster compared to other clusters. Values range from -1 to 1, where higher values indicate better clustering.

```python
from sklearn.metrics import silhouette_score

sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.2f}")
```

### 📏 Davies-Bouldin Index 📏

Calculates the average similarity ratio of each cluster with its most similar cluster. Lower values indicate better clustering.

```python
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_score:.2f}")
```

### 📐 Elbow Method 📐

Helps determine the optimal number of clusters by plotting the sum of squared distances (inertia) against the number of clusters.

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
```

---

## 8. 🛠️📈 Example Project: Customer Segmentation 🛠️📈

Let's apply today's concepts by building a **Customer Segmentation** model using clustering and dimensionality reduction techniques. This project will help businesses understand different customer groups to tailor marketing strategies effectively.

### 📋 Project Overview

**Objective**: Segment customers based on their purchasing behavior and demographics to identify distinct customer groups for targeted marketing.

**Tools**: Python, Scikit-Learn, pandas, Matplotlib, Seaborn

### 📝 Step-by-Step Guide

#### 1. Load and Explore the Dataset

We'll use the **Mall Customers** dataset, which contains information about customers' annual income and spending scores.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')
print(df.head())

# Visualize the data
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df, hue='Gender', palette='Set1')
plt.title('Annual Income vs Spending Score')
plt.show()
```

#### 2. Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# Select relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 3. Clustering with K-Means

```python
from sklearn.cluster import KMeans

# Determine optimal K using Elbow Method
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow
plt.figure(figsize=(8, 4))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
```

Based on the elbow plot, let's choose K=5.

```python
# Initialize and train K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels_

# Add cluster labels to the DataFrame
df['Cluster_KMeans'] = labels_kmeans
```

#### 4. Clustering with DBSCAN

```python
from sklearn.cluster import DBSCAN

# Initialize and train DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
df['Cluster_DBSCAN'] = labels_dbscan
```

#### 5. Dimensionality Reduction with PCA

```python
from sklearn.decomposition import PCA

# Initialize PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame for PCA
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster_KMeans'] = labels_kmeans

# Visualize PCA with K-Means clusters
sns.scatterplot(x='PC1', y='PC2', hue='Cluster_KMeans', data=pca_df, palette='Set2')
plt.title('PCA of Customer Segments (K-Means)')
plt.show()
```

#### 6. Visualization with t-SNE

```python
from sklearn.manifold import TSNE

# Initialize t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_results = tsne.fit_transform(X_scaled)

# Create a DataFrame for t-SNE
tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
tsne_df['Cluster_KMeans'] = labels_kmeans

# Visualize t-SNE with K-Means clusters
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cluster_KMeans', data=tsne_df, palette='coolwarm')
plt.title('t-SNE of Customer Segments (K-Means)')
plt.show()
```

#### 7. Evaluating Clustering Performance

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Silhouette Score for K-Means
sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
print(f"Silhouette Score for K-Means: {sil_kmeans:.2f}")

# Silhouette Score for DBSCAN
sil_dbscan = silhouette_score(X_scaled, labels_dbscan)
print(f"Silhouette Score for DBSCAN: {sil_dbscan:.2f}")

# Davies-Bouldin Index for K-Means
db_kmeans = davies_bouldin_score(X_scaled, labels_kmeans)
print(f"Davies-Bouldin Index for K-Means: {db_kmeans:.2f}")

# Davies-Bouldin Index for DBSCAN
db_dbscan = davies_bouldin_score(X_scaled, labels_dbscan)
print(f"Davies-Bouldin Index for DBSCAN: {db_dbscan:.2f}")
```

---

## 9. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 5** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, you explored **Unsupervised Learning**, mastering clustering algorithms like K-Means, Hierarchical Clustering, and DBSCAN, as well as dimensionality reduction techniques such as PCA and t-SNE. You implemented these techniques using Scikit-Learn and applied them to a real-world customer segmentation project, gaining valuable insights into your data.

### 🔮 What’s Next?

- **Day 6: Advanced Feature Engineering**: Master techniques to create and select features that enhance model performance.
- **Day 7: Ensemble Methods**: Explore ensemble techniques like Bagging, Boosting, and Stacking.
- **Day 8: Model Deployment with Scikit-Learn**: Learn how to deploy your models into production environments.
- **Day 9: Time Series Analysis**: Explore techniques for analyzing and forecasting time-dependent data.
- **Days 10-90: Specialized Topics and Projects**: Engage in specialized topics and comprehensive projects to solidify your expertise.

### 📝 Tips for Success

- **Practice Regularly**: Apply the concepts through exercises and real-world projects.
- **Engage with the Community**: Join forums, attend webinars, and collaborate with peers.
- **Stay Curious**: Continuously explore new features and updates in Scikit-Learn.
- **Document Your Work**: Keep a detailed journal of your learning progress and projects.

Keep up the great work, and stay motivated as you continue your journey to mastering Scikit-Learn and machine learning! 🚀📚


---

# 📜 Summary of Day 5 📜

- **🧠 Introduction to Unsupervised Learning**: Gained a foundational understanding of unsupervised learning concepts and their applications.
- **📊 Clustering Algorithms**: Explored K-Means, Hierarchical Clustering, and DBSCAN, understanding their strengths and use-cases.
- **📉 Dimensionality Reduction Techniques**: Learned about PCA and t-SNE for reducing data dimensionality and enhancing data visualization.
- **🛠️ Implementing Clustering and Dimensionality Reduction with Scikit-Learn**: Practiced building and visualizing clusters and reducing dimensionality using Scikit-Learn.
- **📈 Model Evaluation for Unsupervised Learning**: Mastered evaluation metrics including Silhouette Score, Davies-Bouldin Index, and the Elbow Method.
- **🛠️📈 Example Project: Customer Segmentation**: Developed a customer segmentation project, applying clustering and dimensionality reduction techniques to uncover hidden patterns and groupings in customer data.
  
This structured approach ensures that you build a strong foundation in unsupervised learning techniques, preparing you for more advanced machine learning topics in the upcoming days. Continue experimenting with the provided code examples, and don't hesitate to explore additional resources to deepen your understanding.

**Happy Learning! 🎉**
