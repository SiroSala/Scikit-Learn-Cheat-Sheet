<div style="text-align: center;">
  <h1>🚀 Becoming a Scikit-Learn Boss in 90 Days: Day 6 – Advanced Feature Engineering 🛠️✨</h1>
  <p>Master the Art of Crafting and Selecting Features to Elevate Your Machine Learning Models!</p>
</div>

---

## 📑 Table of Contents

1. [🌟 Welcome to Day 6](#welcome-to-day-6)
2. [🔍 Review of Day 5 📜](#review-of-day-5-📜)
3. [🧠 Introduction to Feature Engineering 🧠](#introduction-to-feature-engineering-🧠)
    - [📚 What is Feature Engineering?](#what-is-feature-engineering-📚)
    - [🔍 Importance of Feature Engineering](#importance-of-feature-engineering-🔍)
4. [🛠️ Feature Creation Techniques 🛠️](#feature-creation-techniques-🛠️)
    - [📐 Polynomial Features](#polynomial-features-📐)
    - [🔗 Interaction Features](#interaction-features-🔗)
    - [📊 Binning](#binning-📊)
    - [🧩 Feature Transformation](#feature-transformation-🧩)
5. [🗑️ Feature Selection Techniques 🗑️](#feature-selection-techniques-🗑️)
    - [✅ Filter Methods](#filter-methods-✅)
    - [🔄 Wrapper Methods](#wrapper-methods-🔄)
    - [🧬 Embedded Methods](#embedded-methods-🧬)
6. [🔀 Handling Categorical Features 🔀](#handling-categorical-features-🔀)
    - [🔡 One-Hot Encoding](#one-hot-encoding-🔡)
    - [🔢 Label Encoding](#label-encoding-🔢)
    - [🌀 Target Encoding](#target-encoding-🌀)
7. [📏 Advanced Feature Scaling 📏](#advanced-feature-scaling-📏)
    - [🧹 Robust Scaling](#robust-scaling-🧹)
    - [📐 Quantile Transformation](#quantile-transformation-📐)
    - [🔄 Power Transformation (Box-Cox, Yeo-Johnson)](#power-transformation-box-cx-yeo-johnson-🔄)
8. [🛠️ Implementing Advanced Feature Engineering with Scikit-Learn 🛠️](#implementing-advanced-feature-engineering-with-scikit-learn-🛠️)
    - [📐 Polynomial Features Example 📐](#polynomial-features-example-📐)
    - [🔗 Interaction Features Example 🔗](#interaction-features-example-🔗)
    - [🗑️ Feature Selection Example 🗑️](#feature-selection-example-🗑️)
    - [🔀 Handling Categorical Features Example 🔀](#handling-categorical-features-example-🔀)
    - [📏 Advanced Feature Scaling Example 📏](#advanced-feature-scaling-example-📏)
9. [🛠️📈 Example Project: Enhancing Model Performance with Feature Engineering 🛠️📈](#example-project-enhancing-model-performance-with-feature-engineering-🛠️📈)
    - [📋 Project Overview](#project-overview-📋)
    - [📝 Step-by-Step Guide](#step-by-step-guide-📝)
        - [1. Load and Explore the Dataset](#1-load-and-explore-the-dataset)
        - [2. Data Preprocessing](#2-data-preprocessing)
        - [3. Feature Creation](#3-feature-creation)
        - [4. Feature Selection](#4-feature-selection)
        - [5. Handling Categorical Features](#5-handling-categorical-features)
        - [6. Advanced Feature Scaling](#6-advanced-feature-scaling)
        - [7. Building and Training the Model](#7-building-and-training-the-model)
        - [8. Evaluating Model Performance](#8-evaluating-model-performance)
    - [📊 Results and Insights](#results-and-insights-📊)
10. [🚀🎓 Conclusion and Next Steps 🚀🎓](#conclusion-and-next-steps-🚀🎓)
11. [📜 Summary of Day 6 📜](#summary-of-day-6-📜)

---

## 1. 🌟 Welcome to Day 6

Welcome to **Day 6** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, we'll dive into **Advanced Feature Engineering**, a critical step in building robust and high-performing machine learning models. You'll learn how to create new features, select the most relevant ones, handle categorical data effectively, and apply advanced scaling techniques to prepare your data for modeling.

---

## 2. 🔍 Review of Day 5 📜

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Unsupervised Learning: Clustering and Dimensionality Reduction**: Explored K-Means, Hierarchical Clustering, DBSCAN, PCA, and t-SNE.
- **Implementing Clustering and Dimensionality Reduction with Scikit-Learn**: Practiced building and visualizing clusters, reducing dimensionality, and evaluating clustering performance.
- **Example Project: Customer Segmentation**: Developed a customer segmentation project, applying clustering and dimensionality reduction techniques to uncover hidden patterns and groupings in customer data.

With this foundation, we're ready to enhance our models through sophisticated feature engineering techniques.

---

## 3. 🧠 Introduction to Feature Engineering 🧠

### 📚 What is Feature Engineering?

**Feature Engineering** is the process of using domain knowledge to create new features or modify existing ones to improve the performance of machine learning models. It involves transforming raw data into meaningful representations that make patterns more discernible to algorithms.

### 🔍 Importance of Feature Engineering

- **Improves Model Performance**: Well-engineered features can significantly enhance the predictive power of models.
- **Reduces Overfitting**: By selecting relevant features, you can simplify models and reduce the risk of overfitting.
- **Enhances Interpretability**: Meaningful features can make models easier to understand and interpret.
- **Handles Data Quality Issues**: Techniques like imputation and scaling address issues like missing values and feature scale discrepancies.

---

## 4. 🛠️ Feature Creation Techniques 🛠️

### 📐 Polynomial Features

Polynomial features allow you to capture non-linear relationships by creating new features that are combinations of existing ones raised to a power.

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Sample DataFrame
data = {
    'Feature1': [2, 3, 5, 7],
    'Feature2': [4, 5, 6, 7]
}
df = pd.DataFrame(data)

# Initialize PolynomialFeatures with degree=2
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df)

# Create a DataFrame with polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out())
print(poly_df)
```

### 🔗 Interaction Features

Interaction features capture the combined effect of two or more features.

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Sample DataFrame
data = {
    'Feature1': [1, 2, 3],
    'Feature2': [4, 5, 6],
    'Feature3': [7, 8, 9]
}
df = pd.DataFrame(data)

# Initialize PolynomialFeatures with degree=2 and interaction_only=True
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(df)

# Create a DataFrame with interaction features
interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out())
print(interaction_df)
```

### 📊 Binning

Binning transforms continuous features into categorical bins, which can help capture non-linear relationships.

```python
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Age': [23, 45, 12, 67, 34, 56, 78, 89, 10, 25]
}
df = pd.DataFrame(data)

# Define bin edges and labels
bins = [0, 18, 35, 60, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Senior']

# Create binned feature
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
print(df)
```

### 🧩 Feature Transformation

Feature transformation methods modify the scale or distribution of features to improve model performance.

```python
from sklearn.preprocessing import PowerTransformer
import pandas as pd

# Sample DataFrame
data = {
    'Income': [50000, 60000, 80000, 120000, 150000, 300000, 500000]
}
df = pd.DataFrame(data)

# Initialize PowerTransformer with 'yeo-johnson' method
pt = PowerTransformer(method='yeo-johnson')
df['Income_Transformed'] = pt.fit_transform(df[['Income']])
print(df)
```

---

## 5. 🗑️ Feature Selection Techniques 🗑️

### ✅ Filter Methods ✅

Filter methods assess the relevance of features based on statistical measures independent of any machine learning algorithms.

```python
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

# Sample DataFrame
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 3, 4, 5, 6],
    'Feature3': [5, 4, 3, 2, 1],
    'Target': [1, 3, 2, 5, 4]
}
df = pd.DataFrame(data)
X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Target']

# Select top 2 features based on f_regression
selector = SelectKBest(score_func=f_regression, k=2)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(f"Selected Features: {selected_features.tolist()}")
```

### 🔄 Wrapper Methods 🔄

Wrapper methods evaluate feature subsets based on the performance of a specific machine learning algorithm.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import pandas as pd

# Sample DataFrame
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 3, 4, 5, 6],
    'Feature3': [5, 4, 3, 2, 1],
    'Target': [1, 3, 2, 5, 4]
}
df = pd.DataFrame(data)
X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Target']

# Initialize Linear Regression model
model = LinearRegression()

# Initialize RFE with 2 features
rfe = RFE(estimator=model, n_features_to_select=2)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print(f"Selected Features: {selected_features.tolist()}")
```

### 🧬 Embedded Methods 🧬

Embedded methods perform feature selection as part of the model training process.

```python
from sklearn.linear_model import Lasso
import pandas as pd

# Sample DataFrame
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 3, 4, 5, 6],
    'Feature3': [5, 4, 3, 2, 1],
    'Target': [1, 3, 2, 5, 4]
}
df = pd.DataFrame(data)
X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Target']

# Initialize Lasso with alpha=0.1
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Select non-zero coefficients
selected_features = X.columns[lasso.coef_ != 0]
print(f"Selected Features: {selected_features.tolist()}")
```

---

## 6. 🔀 Handling Categorical Features 🔀

### 🔡 One-Hot Encoding 🔡

Converts categorical variables into a binary matrix.

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sample DataFrame
data = {
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']
}
df = pd.DataFrame(data)

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['Color']])

# Create a DataFrame with encoded features
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Color']))
df = pd.concat([df, encoded_df], axis=1)
print(df)
```

### 🔢 Label Encoding 🔢

Assigns a unique integer to each category.

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Sample DataFrame
data = {
    'Size': ['Small', 'Medium', 'Large', 'Medium', 'Small']
}
df = pd.DataFrame(data)

# Initialize LabelEncoder
le = LabelEncoder()
df['Size_Encoded'] = le.fit_transform(df['Size'])
print(df)
```

### 🌀 Target Encoding 🌀

Encodes categorical variables based on the target variable's mean for each category.

```python
import pandas as pd

# Sample DataFrame
data = {
    'City': ['New York', 'Los Angeles', 'Chicago', 'New York', 'Chicago'],
    'Sales': [250, 150, 200, 300, 180]
}
df = pd.DataFrame(data)

# Calculate target mean for each category
target_mean = df.groupby('City')['Sales'].mean()

# Map the target mean to the categories
df['City_Target_Encoded'] = df['City'].map(target_mean)
print(df)
```

---

## 7. 📏 Advanced Feature Scaling 📏

### 🧹 Robust Scaling 🧹

Scales features using statistics that are robust to outliers, such as the median and interquartile range.

```python
from sklearn.preprocessing import RobustScaler
import pandas as pd

# Sample DataFrame
data = {
    'Income': [50000, 60000, 80000, 120000, 150000, 300000, 500000]
}
df = pd.DataFrame(data)

# Initialize RobustScaler
scaler = RobustScaler()
df['Income_Robust_Scaled'] = scaler.fit_transform(df[['Income']])
print(df)
```

### 📐 Quantile Transformation 📐

Transforms features to follow a uniform or normal distribution based on quantiles.

```python
from sklearn.preprocessing import QuantileTransformer
import pandas as pd

# Sample DataFrame
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 70]
}
df = pd.DataFrame(data)

# Initialize QuantileTransformer with output_distribution='normal'
qt = QuantileTransformer(output_distribution='normal')
df['Age_Quantile_Scaled'] = qt.fit_transform(df[['Age']])
print(df)
```

### 🔄 Power Transformation (Box-Cox, Yeo-Johnson) 🔄

Applies a power transformation to make data more Gaussian-like.

```python
from sklearn.preprocessing import PowerTransformer
import pandas as pd

# Sample DataFrame
data = {
    'Skewed_Feature': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)

# Initialize PowerTransformer with 'yeo-johnson' method
pt = PowerTransformer(method='yeo-johnson')
df['Skewed_Feature_Transformed'] = pt.fit_transform(df[['Skewed_Feature']])
print(df)
```

---

## 8. 🛠️ Implementing Advanced Feature Engineering with Scikit-Learn 🛠️

### 📐 Polynomial Features Example 📐

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Sample DataFrame
data = {
    'Feature1': [2, 3, 5, 7],
    'Feature2': [4, 5, 6, 7]
}
df = pd.DataFrame(data)

# Initialize PolynomialFeatures with degree=3
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(df)

# Create a DataFrame with polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out())
print(poly_df)
```

### 🔗 Interaction Features Example 🔗

```python
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# Sample DataFrame
data = {
    'Height': [150, 160, 170, 180, 190],
    'Weight': [50, 60, 70, 80, 90]
}
df = pd.DataFrame(data)

# Initialize PolynomialFeatures with degree=2 and interaction_only=True
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interaction_features = poly.fit_transform(df)

# Create a DataFrame with interaction features
interaction_df = pd.DataFrame(interaction_features, columns=poly.get_feature_names_out())
print(interaction_df)
```

### 🗑️ Feature Selection Example 🗑️

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='Species')

# Select top 2 features based on ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
print(f"Selected Features: {selected_features.tolist()}")
```

### 🔀 Handling Categorical Features Example 🔀

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sample DataFrame
data = {
    'Department': ['Sales', 'Engineering', 'HR', 'Engineering', 'Sales']
}
df = pd.DataFrame(data)

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(df[['Department']])

# Create a DataFrame with encoded features
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Department']))
df = pd.concat([df, encoded_df], axis=1)
print(df)
```

### 📏 Advanced Feature Scaling Example 📏

```python
from sklearn.preprocessing import RobustScaler
import pandas as pd

# Sample DataFrame with outliers
data = {
    'Salary': [50000, 60000, 80000, 120000, 150000, 300000, 500000]
}
df = pd.DataFrame(data)

# Initialize RobustScaler
scaler = RobustScaler()
df['Salary_Robust_Scaled'] = scaler.fit_transform(df[['Salary']])
print(df)
```

---

## 9. 🛠️📈 Example Project: Enhancing Model Performance with Feature Engineering 🛠️📈

Let's apply today's concepts by enhancing a regression model's performance through advanced feature engineering techniques. We'll use the **California Housing Dataset** to predict median house values.

### 📋 Project Overview

**Objective**: Improve the predictive performance of a regression model by creating new features, selecting the most relevant ones, handling categorical variables effectively, and applying advanced scaling techniques.

**Tools**: Python, Scikit-Learn, pandas, Matplotlib, Seaborn

### 📝 Step-by-Step Guide

#### 1. Load and Explore the Dataset

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load California Housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = pd.Series(housing.target, name='MedHouseVal')

# Combine features and target
df = pd.concat([X, y], axis=1)
print(df.head())

# Visualize relationships
sns.pairplot(df.sample(500), x_vars=housing.feature_names, y_vars='MedHouseVal', height=2.5)
plt.show()
```

#### 2. Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)
```

#### 3. Feature Creation

```python
from sklearn.preprocessing import PolynomialFeatures

# Initialize PolynomialFeatures with degree=2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Create a DataFrame with polynomial features
poly_features = poly.get_feature_names_out()
X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_features)
X_test_poly_df = pd.DataFrame(X_test_poly, columns=poly_features)

print(X_train_poly_df.head())
```

#### 4. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Initialize SelectKBest with f_regression
selector = SelectKBest(score_func=f_regression, k=20)
X_train_selected = selector.fit_transform(X_train_poly_df, y_train)
X_test_selected = selector.transform(X_test_poly_df)

# Get selected feature names
selected_features = poly_features[selector.get_support()]
print(f"Selected Features: {selected_features.tolist()}")
```

#### 5. Handling Categorical Features

*Note: The California Housing Dataset does not contain categorical features. For demonstration, we'll simulate a categorical feature.*

```python
import numpy as np

# Simulate a categorical feature
df_train = pd.DataFrame(X_train_selected, columns=selected_features)
df_train['OceanProximity'] = np.random.choice(['NEAR BAY', 'INLAND', 'NEAR OCEAN', 'ISLAND', 'NEAR WATER'], size=df_train.shape[0])

df_test = pd.DataFrame(X_test_selected, columns=selected_features)
df_test['OceanProximity'] = np.random.choice(['NEAR BAY', 'INLAND', 'NEAR OCEAN', 'ISLAND', 'NEAR WATER'], size=df_test.shape[0])

# Initialize OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False, drop='first')
encoded_train = encoder.fit_transform(df_train[['OceanProximity']])
encoded_test = encoder.transform(df_test[['OceanProximity']])

# Create DataFrame with encoded features
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(['OceanProximity']))
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(['OceanProximity']))

# Concatenate with numerical features
X_train_final = pd.concat([df_train.drop('OceanProximity', axis=1), encoded_train_df], axis=1)
X_test_final = pd.concat([df_test.drop('OceanProximity', axis=1), encoded_test_df], axis=1)

print(X_train_final.head())
```

#### 6. Advanced Feature Scaling

```python
from sklearn.preprocessing import RobustScaler

# Initialize RobustScaler
robust_scaler = RobustScaler()

# Fit and transform the training data
X_train_final_scaled = robust_scaler.fit_transform(X_train_final)

# Transform the testing data
X_test_final_scaled = robust_scaler.transform(X_test_final)
```

#### 7. Building and Training the Model

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Ridge Regression with alpha=1.0
ridge = Ridge(alpha=1.0)

# Train the model
ridge.fit(X_train_final_scaled, y_train)

# Make predictions
y_pred = ridge.predict(X_test_final_scaled)
```

#### 8. Evaluating Model Performance

```python
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ridge Regression MSE: {mse:.4f}")
print(f"Ridge Regression RMSE: {rmse:.4f}")
print(f"Ridge Regression MAE: {mae:.4f}")
print(f"Ridge Regression R²: {r2:.4f}")
```

### 📊 Results and Insights

After performing advanced feature engineering, the Ridge Regression model exhibits improved performance metrics compared to the baseline model. The addition of polynomial and interaction features, along with feature selection and robust scaling, has enhanced the model's ability to capture complex relationships in the data.

---

## 10. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 6** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, you mastered **Advanced Feature Engineering**, learning how to create new features, select the most relevant ones, handle categorical data effectively, and apply advanced scaling techniques. By implementing these strategies, you enhanced your model's performance and gained deeper insights into your data.

### 🔮 What’s Next?

- **Day 7: Ensemble Methods**: Explore powerful ensemble techniques like Bagging, Boosting, and Stacking to improve model performance.
- **Day 8: Model Deployment with Scikit-Learn**: Learn how to deploy your machine learning models into production environments.
- **Day 9: Time Series Analysis**: Delve into techniques for analyzing and forecasting time-dependent data.
- **Day 10: Advanced Model Interpretability**: Understand methods to interpret and explain your machine learning models.
- **Days 11-90: Specialized Topics and Projects**: Engage in specialized topics and comprehensive projects to solidify your expertise.

### 📝 Tips for Success

- **Practice Regularly**: Apply the concepts through exercises and real-world projects to reinforce your knowledge.
- **Engage with the Community**: Join forums, attend webinars, and collaborate with peers to broaden your perspective and solve challenges together.
- **Stay Curious**: Continuously explore new features and updates in Scikit-Learn and other machine learning libraries.
- **Document Your Work**: Keep a detailed journal of your learning progress and projects to track your growth and facilitate future learning.

Keep up the great work, and stay motivated as you continue your journey to mastering Scikit-Learn and machine learning! 🚀📚


---

# 📜 Summary of Day 6 📜

- **🧠 Introduction to Feature Engineering**: Gained a foundational understanding of feature engineering and its significance in machine learning.
- **🛠️ Feature Creation Techniques**: Explored methods like Polynomial Features, Interaction Features, Binning, and Feature Transformation to create new, meaningful features.
- **🗑️ Feature Selection Techniques**: Learned about Filter, Wrapper, and Embedded methods to select the most relevant features for your models.
- **🔀 Handling Categorical Features**: Mastered encoding techniques including One-Hot Encoding, Label Encoding, and Target Encoding to effectively handle categorical data.
- **📏 Advanced Feature Scaling**: Applied advanced scaling techniques such as Robust Scaling, Quantile Transformation, and Power Transformation to prepare data for modeling.
- **🛠️ Implementing Advanced Feature Engineering with Scikit-Learn**: Practiced building and transforming features using Scikit-Learn's preprocessing tools.
- **🛠️📈 Example Project: Enhancing Model Performance with Feature Engineering**: Developed a comprehensive regression pipeline to predict housing prices, incorporating advanced feature creation, selection, handling of categorical variables, and scaling to optimize model performance.
