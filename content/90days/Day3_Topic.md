<div style="text-align: center;">
  <h1>🚀 Becoming a Scikit-Learn Boss in 90 Days: Day 3 – Supervised Learning: Regression Algorithms 📈🔍</h1>
  <p>Dive Deeper into Regression Techniques to Enhance Your Predictive Models!</p>
</div>

---

## 📑 Table of Contents

1. [🌟 Welcome to Day 3](#welcome-to-day-3)
2. [🔍 Review of Day 2 📜](#review-of-day-2-📜)
3. [🧠 Introduction to Supervised Learning: Regression 🧠](#introduction-to-supervised-learning-regression-🧠)
    - [📚 What is Regression?](#what-is-regression-📚)
    - [🔍 Types of Regression Problems](#types-of-regression-problems-🔍)
4. [📊 Regression Algorithms](#regression-algorithms-📊)
    - [📈 Linear Regression 📈](#linear-regression-📈)
    - [🪓 Ridge Regression 🪓](#ridge-regression-🪓)
    - [✂️ Lasso Regression ✂️](#lasso-regression-✂️)
    - [🔗 Elastic Net 🔗](#elastic-net-🔗)
5. [🛠️ Implementing Regression Algorithms with Scikit-Learn 🛠️](#implementing-regression-algorithms-with-scikit-learn-🛠️)
    - [📈 Linear Regression Example 📈](#linear-regression-example-📈)
    - [🪓 Ridge Regression Example 🪓](#ridge-regression-example-🪓)
    - [✂️ Lasso Regression Example ✂️](#lasso-regression-example-✂️)
    - [🔗 Elastic Net Example 🔗](#elastic-net-example-🔗)
6. [📈 Model Evaluation for Regression](#model-evaluation-for-regression-📈)
    - [📉 Mean Squared Error (MSE) 📉](#mean-squared-error-mse-📉)
    - [📏 Root Mean Squared Error (RMSE) 📏](#root-mean-squared-error-rmse-📏)
    - [✍️ Mean Absolute Error (MAE) ✍️](#mean-absolute-error-mae-✍️)
    - [📈 R-squared (R²) 📈](#r-squared-r²-📈)
7. [🛠️📈 Example Project: Housing Price Prediction](#example-project-housing-price-prediction-🛠️📈)
8. [🚀🎓 Conclusion and Next Steps](#conclusion-and-next-steps-🚀🎓)
9. [📜 Summary of Day 3 📜](#summary-of-day-3-📜)

---

## 1. 🌟 Welcome to Day 3

Welcome to **Day 3** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, we'll explore **Supervised Learning** with a focus on **Regression Algorithms**. You'll learn about different regression techniques, implement them using Scikit-Learn, and evaluate their performance to build more accurate predictive models.

---

## 2. 🔍 Review of Day 2 📜

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Supervised Learning: Classification Algorithms**: Explored Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).
- **Implementing Classification Algorithms with Scikit-Learn**: Built, trained, and evaluated various classification models.
- **Model Evaluation for Classification**: Learned about accuracy, precision, recall, F1-score, confusion matrix, and ROC curves.
- **Example Project: Advanced Iris Classification**: Developed a comprehensive classification pipeline using multiple algorithms to classify Iris species and compared their performance.

With this foundation, we're ready to delve into regression techniques that will enhance your ability to make continuous predictions.

---

## 3. 🧠 Introduction to Supervised Learning: Regression 🧠

### 📚 What is Regression?

**Regression** is a type of supervised learning where the goal is to predict a continuous target variable based on one or more predictor variables. Unlike classification, which deals with categorical outcomes, regression focuses on estimating numerical values.

### 🔍 Types of Regression Problems

- **Simple Linear Regression**: Predicting a target variable using a single feature.
- **Multiple Linear Regression**: Predicting a target variable using multiple features.
- **Regularized Regression**: Techniques like Ridge, Lasso, and Elastic Net that add penalties to prevent overfitting.
- **Polynomial Regression**: Extending linear models to capture non-linear relationships.

---

## 4. 📊 Regression Algorithms 📊

### 📈 Linear Regression 📈

A foundational regression technique that models the relationship between the target variable and one or more features by fitting a linear equation.

### 🪓 Ridge Regression 🪓

A regularized version of linear regression that adds an L2 penalty to the loss function to prevent overfitting by shrinking the coefficients.

### ✂️ Lasso Regression ✂️

Another regularized regression method that adds an L1 penalty, which can shrink some coefficients to zero, effectively performing feature selection.

### 🔗 Elastic Net 🔗

Combines both L1 and L2 regularization penalties, balancing the benefits of Ridge and Lasso regression.

---

## 5. 🛠️ Implementing Regression Algorithms with Scikit-Learn 🛠️

### 📈 Linear Regression Example 📈

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
linear_reg = LinearRegression()

# Train the model
linear_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_linear = linear_reg.predict(X_test_scaled)

# Evaluate the model
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"Linear Regression MSE: {mse_linear:.2f}")
print(f"Linear Regression R²: {r2_linear:.2f}")
```

### 🪓 Ridge Regression Example 🪓

```python
from sklearn.linear_model import Ridge

# Initialize the model with alpha=1.0
ridge_reg = Ridge(alpha=1.0)

# Train the model
ridge_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_ridge = ridge_reg.predict(X_test_scaled)

# Evaluate the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"Ridge Regression MSE: {mse_ridge:.2f}")
print(f"Ridge Regression R²: {r2_ridge:.2f}")
```

### ✂️ Lasso Regression Example ✂️

```python
from sklearn.linear_model import Lasso

# Initialize the model with alpha=0.1
lasso_reg = Lasso(alpha=0.1)

# Train the model
lasso_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lasso = lasso_reg.predict(X_test_scaled)

# Evaluate the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso Regression MSE: {mse_lasso:.2f}")
print(f"Lasso Regression R²: {r2_lasso:.2f}")
```

### 🔗 Elastic Net Example 🔗

```python
from sklearn.linear_model import ElasticNet

# Initialize the model with alpha=0.1 and l1_ratio=0.5
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train the model
elastic_net.fit(X_train_scaled, y_train)

# Make predictions
y_pred_elastic = elastic_net.predict(X_test_scaled)

# Evaluate the model
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)
print(f"Elastic Net MSE: {mse_elastic:.2f}")
print(f"Elastic Net R²: {r2_elastic:.2f}")
```

---

## 6. 📈 Model Evaluation for Regression 📈

### 📉 Mean Squared Error (MSE) 📉

Measures the average of the squares of the errors, providing an indication of the quality of the estimator.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

### 📏 Root Mean Squared Error (RMSE) 📏

The square root of MSE, representing the standard deviation of the residuals.

```python
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse:.2f}")
```

### ✍️ Mean Absolute Error (MAE) ✍️

Calculates the average of the absolute errors, providing a straightforward measure of prediction accuracy.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
```

### 📈 R-squared (R²) 📈

Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")
```

---

## 7. 🛠️📈 Example Project: Housing Price Prediction 🛠️📈

Let's apply today's concepts by building a regression model to predict housing prices using the **California Housing Dataset**.

### 📋 Project Overview

**Objective**: Develop a machine learning pipeline to predict housing prices based on various features such as location, size, and demographics.

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

# Visualize distribution of target variable
sns.histplot(df['MedHouseVal'], bins=50, kde=True)
plt.title('Distribution of Median House Values')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.show()
```

#### 2. Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### 3. Building and Training the Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Initialize models
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
lasso_reg = Lasso(alpha=0.1)
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train models
linear_reg.fit(X_train_scaled, y_train)
ridge_reg.fit(X_train_scaled, y_train)
lasso_reg.fit(X_train_scaled, y_train)
elastic_net.fit(X_train_scaled, y_train)
```

#### 4. Making Predictions and Evaluating the Models

```python
from sklearn.metrics import mean_squared_error, r2_score

models = {
    'Linear Regression': linear_reg,
    'Ridge Regression': ridge_reg,
    'Lasso Regression': lasso_reg,
    'Elastic Net': elastic_net
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} Evaluation Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}\n")
```

#### 5. Comparing Model Performance

```python
import numpy as np

# Initialize a DataFrame to store evaluation metrics
evaluation_df = pd.DataFrame(columns=['Model', 'MSE', 'RMSE', 'MAE', 'R²'])

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    evaluation_df = evaluation_df.append({
        'Model': name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }, ignore_index=True)

print(evaluation_df)

# Visualize the comparison
sns.barplot(x='R²', y='Model', data=evaluation_df, palette='coolwarm')
plt.title('R² Score Comparison of Regression Models')
plt.xlabel('R² Score')
plt.ylabel('Model')
plt.xlim(0, 1)
plt.show()
```

---

## 8. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 3** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, you delved into **Supervised Learning: Regression Algorithms**, mastering techniques like Linear Regression, Ridge, Lasso, and Elastic Net. You implemented these algorithms using Scikit-Learn, evaluated their performance, and applied them to a real-world dataset to predict housing prices.

### 🔮 What’s Next?

- **Day 4: Model Evaluation and Selection**: Learn about cross-validation, hyperparameter tuning, and strategies to select the best model.
- **Day 5: Unsupervised Learning – Clustering and Dimensionality Reduction**: Understand clustering algorithms like K-Means and techniques like PCA.
- **Day 6: Advanced Feature Engineering**: Master techniques to create and select features that enhance model performance.
- **Day 7: Ensemble Methods**: Explore ensemble techniques like Bagging, Boosting, and Stacking.
- **Day 8: Model Deployment with Scikit-Learn**: Learn how to deploy your models into production environments.
- **Days 9-90: Specialized Topics and Projects**: Engage in specialized topics and comprehensive projects to solidify your expertise.

### 📝 Tips for Success

- **Practice Regularly**: Apply the concepts through exercises and real-world projects.
- **Engage with the Community**: Join forums, attend webinars, and collaborate with peers.
- **Stay Curious**: Continuously explore new features and updates in Scikit-Learn.
- **Document Your Work**: Keep a detailed journal of your learning progress and projects.

Keep up the great work, and stay motivated as you continue your journey to mastering Scikit-Learn and machine learning! 🚀📚

---

# 📜 Summary of Day 3 📜

- **🧠 Introduction to Supervised Learning: Regression**: Gained a foundational understanding of regression tasks and their types.
- **📊 Regression Algorithms**: Explored Linear Regression, Ridge Regression, Lasso Regression, and Elastic Net.
- **🛠️ Implementing Regression Algorithms with Scikit-Learn**: Learned how to build, train, and evaluate different regression models using Scikit-Learn.
- **📈 Model Evaluation for Regression**: Mastered evaluation metrics including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
- **🛠️📈 Example Project: Housing Price Prediction**: Developed a comprehensive regression pipeline using multiple algorithms to predict housing prices and compared their performance.

