<div style="text-align: center;">
  <h1>🚀 Becoming a Scikit-Learn Boss in 90 Days: Day 9 – Time Series Analysis 📅📈</h1>
  <p>Master the Techniques to Analyze and Forecast Time-Dependent Data!</p>
</div>

---

## 📑 Table of Contents

1. [🌟 Welcome to Day 9](#welcome-to-day-9)
2. [🔍 Review of Day 8 📜](#review-of-day-8-📜)
3. [🧠 Introduction to Time Series Analysis 🧠](#introduction-to-time-series-analysis-🧠)
    - [📚 What is Time Series Analysis?](#what-is-time-series-analysis-📚)
    - [🔍 Importance of Time Series Analysis](#importance-of-time-series-analysis-🔍)
4. [📊 Time Series Forecasting Techniques 📊](#time-series-forecasting-techniques-📊)
    - [🔄 Moving Averages](#moving-averages-🔄)
    - [📈 Exponential Smoothing](#exponential-smoothing-📈)
    - [📉 Autoregressive (AR) Models](#autoregressive-ar-models-📉)
    - [🔗 Autoregressive Integrated Moving Average (ARIMA)](#autoregressive-integrated-moving-average-arima-🔗)
    - [🧰 Machine Learning Approaches](#machine-learning-approaches-🧰)
        - [📐 Regression-Based Models](#regression-based-models-📐)
        - [🌟 Support Vector Regression (SVR)](#support-vector-regression-svr-🌟)
        - [🛠️ Random Forest Regression](#random-forest-regression-🛠️)
5. [🛠️ Implementing Time Series Analysis with Scikit-Learn 🛠️](#implementing-time-series-analysis-with-scikit-learn-🛠️)
    - [🔄 Preparing Time Series Data](#preparing-time-series-data-🔄)
    - [📐 Feature Engineering for Time Series](#feature-engineering-for-time-series-📐)
    - [📈 Forecasting with Regression Models](#forecasting-with-regression-models-📈)
    - [🌟 Support Vector Regression Example 🌟](#support-vector-regression-example-🌟)
    - [🛠️ Random Forest Regression Example 🛠️](#random-forest-regression-example-🛠️)
6. [📈 Model Evaluation for Time Series 📈](#model-evaluation-for-time-series-📈)
    - [📉 Mean Absolute Error (MAE) 📉]
    - [📏 Mean Squared Error (MSE) 📏]
    - [📐 Root Mean Squared Error (RMSE) 📐]
    - [📈 Mean Absolute Percentage Error (MAPE) 📈]
    - [🔄 Cross-Validation for Time Series](#cross-validation-for-time-series)
7. [🛠️📈 Example Project: Sales Forecasting 🛠️📈](#example-project-sales-forecasting-🛠️📈)
    - [📋 Project Overview](#project-overview-📋)
    - [📝 Step-by-Step Guide](#step-by-step-guide-📝)
        - [1. Load and Explore the Dataset](#1-load-and-explore-the-dataset)
        - [2. Data Preprocessing](#2-data-preprocessing)
        - [3. Feature Engineering](#3-feature-engineering)
        - [4. Building Forecasting Models](#4-building-forecasting-models)
        - [5. Evaluating Model Performance](#5-evaluating-model-performance)
        - [6. Selecting the Best Model](#6-selecting-the-best-model)
        - [7. Making Future Predictions](#7-making-future-predictions)
    - [📊 Results and Insights](#results-and-insights-📊)
8. [🚀🎓 Conclusion and Next Steps 🚀🎓](#conclusion-and-next-steps-🚀🎓)
9. [📜 Summary of Day 9 📜](#summary-of-day-9-📜)

---

## 1. 🌟 Welcome to Day 9

Welcome to **Day 9** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, we'll delve into **Time Series Analysis**, a crucial area for analyzing and forecasting data points collected or recorded at specific time intervals. You'll learn about various time series forecasting techniques, implement them using Scikit-Learn, and apply these methods to real-world datasets to make accurate predictions.

---

## 2. 🔍 Review of Day 8 📜

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Model Deployment**: Learned how to serialize machine learning models using Joblib and Pickle, create RESTful APIs with Flask and FastAPI, and deploy models using advanced tools like MLflow.
- **Steps to Deploy a Model**: Covered the entire deployment pipeline from training and serialization to setting up APIs and monitoring.
- **Example Project**: Successfully deployed a Random Forest Regressor model as a Flask API to predict housing prices, demonstrating real-world applicability.

With this foundation, we're ready to explore the fascinating world of time series analysis, enhancing our ability to work with sequential data and make informed forecasts.

---

## 3. 🧠 Introduction to Time Series Analysis 🧠

### 📚 What is Time Series Analysis? 📚

**Time Series Analysis** involves statistical techniques to model and predict future values based on previously observed values. It is widely used in various domains such as finance, economics, weather forecasting, sales forecasting, and more.

**Key Characteristics of Time Series Data:**

- **Temporal Ordering**: Data points are ordered in time.
- **Seasonality**: Patterns that repeat at regular intervals (e.g., monthly sales).
- **Trend**: Long-term increase or decrease in the data.
- **Stationarity**: Statistical properties like mean and variance remain constant over time.

### 🔍 Importance of Time Series Analysis 🔍

- **Forecasting**: Predict future values (e.g., stock prices, demand forecasting).
- **Anomaly Detection**: Identify unusual patterns or outliers.
- **Seasonal Adjustment**: Remove seasonal effects to better understand underlying trends.
- **Economic Planning**: Aid in making informed business and policy decisions.

---

## 4. 📊 Time Series Forecasting Techniques 📊

### 🔄 Moving Averages 🔄

A technique to smooth out short-term fluctuations and highlight longer-term trends or cycles.

**Types:**

- **Simple Moving Average (SMA)**: Calculates the average of a fixed number of past observations.
- **Weighted Moving Average (WMA)**: Assigns different weights to past observations, giving more importance to recent data.

### 📈 Exponential Smoothing 📈

Applies exponentially decreasing weights to past observations, giving more importance to recent data points.

**Types:**

- **Single Exponential Smoothing**: Suitable for data with no trend or seasonality.
- **Double Exponential Smoothing**: Accounts for trends in the data.
- **Triple Exponential Smoothing (Holt-Winters)**: Handles both trend and seasonality.

### 📉 Autoregressive (AR) Models 📉

Models that use the dependent relationship between an observation and a number of lagged observations.

**AR(p)**: Autoregressive model of order p, where p is the number of lagged observations.

### 🔗 Autoregressive Integrated Moving Average (ARIMA) 🔗

Combines autoregressive and moving average components, with differencing to make the time series stationary.

**Components:**

- **AR (p)**: Autoregressive part.
- **I (d)**: Differencing to achieve stationarity.
- **MA (q)**: Moving average part.

### 🧰 Machine Learning Approaches 🧰

#### 📐 Regression-Based Models 📐

Transform time series forecasting into a regression problem by using lagged values and other time-based features as predictors.

#### 🌟 Support Vector Regression (SVR) 🌟

Uses Support Vector Machines for regression tasks, capable of modeling non-linear relationships.

#### 🛠️ Random Forest Regression 🛠️

An ensemble method that builds multiple decision trees and averages their predictions, robust to overfitting.

---

## 5. 🛠️ Implementing Time Series Analysis with Scikit-Learn 🛠️

### 🔄 Preparing Time Series Data 🔄

1. **Stationarity Check**: Use tests like Augmented Dickey-Fuller (ADF) to check for stationarity.
2. **Differencing**: Apply differencing to remove trends and make the series stationary.
3. **Lag Features**: Create lagged features to capture temporal dependencies.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Sample Time Series Data
data = {
    'Date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'Value': np.random.randn(100).cumsum() + 50
}
df = pd.DataFrame(data)
df.set_index('Date', inplace=True)

# Plot the Time Series
df['Value'].plot(title='Sample Time Series')
plt.show()

# Augmented Dickey-Fuller Test
result = adfuller(df['Value'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
```

### 📐 Feature Engineering for Time Series 📐

Create features such as lagged values, rolling statistics, and time-based features.

```python
# Create Lag Features
df['Lag1'] = df['Value'].shift(1)
df['Lag2'] = df['Value'].shift(2)

# Create Rolling Mean
df['Rolling_Mean_3'] = df['Value'].rolling(window=3).mean()

# Create Time-Based Features
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Weekday'] = df.index.weekday

# Drop NaN Values
df.dropna(inplace=True)
print(df.head())
```

### 📈 Forecasting with Regression Models 📈

Use regression models by treating the problem as predicting the next value based on previous features.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define Features and Target
X = df[['Lag1', 'Lag2', 'Rolling_Mean_3', 'Day', 'Month', 'Weekday']]
y = df['Value']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

### 🌟 Support Vector Regression Example 🌟

```python
from sklearn.svm import SVR

# Initialize SVR with RBF Kernel
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

# Train the Model
svr.fit(X_train, y_train)

# Make Predictions
y_pred_svr = svr.predict(X_test)

# Evaluate the Model
mse_svr = mean_squared_error(y_test, y_pred_svr)
print(f"SVR Mean Squared Error: {mse_svr:.2f}")
```

### 🛠️ Random Forest Regression Example 🛠️

```python
from sklearn.ensemble import RandomForestRegressor

# Initialize Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Model
rf.fit(X_train, y_train)

# Make Predictions
y_pred_rf = rf.predict(X_test)

# Evaluate the Model
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest Mean Squared Error: {mse_rf:.2f}")
```

---

## 6. 📈 Model Evaluation for Time Series 📈

### 📉 Mean Absolute Error (MAE) 📉

Measures the average magnitude of the errors in a set of predictions, without considering their direction.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
```

### 📏 Mean Squared Error (MSE) 📏

Measures the average of the squares of the errors, giving higher weight to larger errors.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

### 📐 Root Mean Squared Error (RMSE) 📐

The square root of MSE, representing the standard deviation of the residuals.

```python
import numpy as np

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")
```

### 📈 Mean Absolute Percentage Error (MAPE) 📈

Measures the accuracy as a percentage, useful for understanding the error relative to the actual values.

```python
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
```

### 🔄 Cross-Validation for Time Series 🔄

Use `TimeSeriesSplit` to perform cross-validation without shuffling the data, preserving the temporal order.

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Perform Cross-Validation with Linear Regression
scores = cross_val_score(LinearRegression(), X, y, cv=tscv, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE Scores: {-scores}")
print(f"Average MSE: {-scores.mean():.2f}")
```

---

## 7. 🛠️📈 Example Project: Sales Forecasting 🛠️📈

### 📋 Project Overview

**Objective**: Develop a machine learning pipeline to forecast daily sales based on historical sales data and related features.

**Tools**: Python, Scikit-Learn, pandas, NumPy, Matplotlib, Seaborn

### 📝 Step-by-Step Guide

#### 1. Load and Explore the Dataset

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Sales Dataset
df = pd.read_csv('daily_sales.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
print(df.head())

# Plot Sales Over Time
df['Sales'].plot(title='Daily Sales Over Time')
plt.show()

# Check for Seasonality and Trend
sns.lineplot(x=df.index, y='Sales', data=df)
plt.title('Sales Trend')
plt.show()
```

#### 2. Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Handle Missing Values if any
df.fillna(method='ffill', inplace=True)

# Create Lag Features
df['Lag1'] = df['Sales'].shift(1)
df['Lag2'] = df['Sales'].shift(2)

# Create Rolling Features
df['Rolling_Mean_7'] = df['Sales'].rolling(window=7).mean()
df['Rolling_STD_7'] = df['Sales'].rolling(window=7).std()

# Create Time-Based Features
df['Day'] = df.index.day
df['Month'] = df.index.month
df['Weekday'] = df.index.weekday

# Drop NaN Values
df.dropna(inplace=True)
print(df.head())
```

#### 3. Feature Engineering

```python
from sklearn.preprocessing import PolynomialFeatures

# Initialize PolynomialFeatures with degree=2
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Lag1', 'Lag2', 'Rolling_Mean_7', 'Rolling_STD_7']])

# Create a DataFrame with polynomial features
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out())
df = pd.concat([df, poly_df], axis=1)
print(df.head())
```

#### 4. Building Forecasting Models

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Define Features and Target
X = df.drop('Sales', axis=1)
y = df['Sales']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize Models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
svr = SVR(kernel='rbf')

# Train Models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
svr.fit(X_train, y_train)
```

#### 5. Evaluating Model Performance

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make Predictions
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svr = svr.predict(X_test)

# Calculate Metrics
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{model_name} Performance:")
    print(f"  MSE: {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.2f}")
    print(f"  MAPE: {mape:.2f}%\n")

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")
evaluate_model(y_test, y_pred_svr, "Support Vector Regressor")
```

#### 6. Selecting the Best Model

Based on the evaluation metrics, choose the model with the best performance (e.g., lowest MSE and RMSE, highest R²).

#### 7. Making Future Predictions

```python
# Assume we have new data for the next day
new_data = {
    'Lag1': [y_test.iloc[-1]],
    'Lag2': [y_test.iloc[-2]],
    'Rolling_Mean_7': [df['Rolling_Mean_7'].iloc[-1]],
    'Rolling_STD_7': [df['Rolling_STD_7'].iloc[-1]],
    'Day': [df.index[-1].day + 1],
    'Month': [df.index[-1].month],
    'Weekday': [(df.index[-1].weekday() + 1) % 7]
}

new_df = pd.DataFrame(new_data)

# Create Polynomial Features
new_poly = poly.transform(new_df[['Lag1', 'Lag2', 'Rolling_Mean_7', 'Rolling_STD_7']])
new_poly_df = pd.DataFrame(new_poly, columns=poly.get_feature_names_out())
new_df = pd.concat([new_df, new_poly_df], axis=1)

# Make Prediction with the Best Model (e.g., Random Forest)
best_model = rf
future_prediction = best_model.predict(new_df)
print(f"Future Sales Prediction: {future_prediction[0]:.2f}")
```

### 📊 Results and Insights

After implementing advanced feature engineering and evaluating multiple models, the **Random Forest Regressor** outperformed the other models with the lowest MSE and highest R² score. This indicates its robustness and ability to capture complex relationships in the sales data, making it the best choice for forecasting future sales.

---

## 8. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 9** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, you mastered **Time Series Analysis**, learning how to preprocess time-dependent data, engineer relevant features, implement forecasting models using regression-based approaches, and evaluate their performance. By working through the sales forecasting example project, you gained hands-on experience in predicting future sales based on historical data.

### 🔮 What’s Next?

- **Day 10: Advanced Model Interpretability**: Understand methods to interpret and explain your machine learning models.
- **Days 11-90: Specialized Topics and Projects**: Engage in specialized topics such as Natural Language Processing, Computer Vision, deep learning integration, and comprehensive projects to solidify your expertise.
- **Continuous Learning**: Explore advanced libraries and tools that complement Scikit-Learn, such as TensorFlow, Keras, and PyTorch for deep learning applications.

### 📝 Tips for Success

- **Practice Regularly**: Apply the concepts through exercises and real-world projects to reinforce your knowledge.
- **Engage with the Community**: Join forums, attend webinars, and collaborate with peers to broaden your perspective and solve challenges together.
- **Stay Curious**: Continuously explore new features and updates in Scikit-Learn and other machine learning libraries.
- **Document Your Work**: Keep a detailed journal of your learning progress and projects to track your growth and facilitate future learning.

Keep up the great work, and stay motivated as you continue your journey to mastering Scikit-Learn and machine learning! 🚀📚

---

# 📜 Summary of Day 9 📜

- **🧠 Introduction to Time Series Analysis**: Gained a foundational understanding of time series data and its unique characteristics.
- **📊 Time Series Forecasting Techniques**: Explored various forecasting methods including Moving Averages, Exponential Smoothing, AR Models, ARIMA, and Machine Learning approaches like SVR and Random Forest Regression.
- **🛠️ Implementing Time Series Analysis with Scikit-Learn**: Learned how to prepare time series data, engineer relevant features, and build forecasting models using regression-based techniques.
- **📈 Model Evaluation for Time Series**: Mastered evaluation metrics such as MAE, MSE, RMSE, MAPE, and appropriate cross-validation techniques for time series data.
- **🛠️📈 Example Project: Sales Forecasting**: Developed a comprehensive sales forecasting pipeline, implemented multiple forecasting models, evaluated their performance, and selected the best model for future predictions.
  
This structured approach ensures that you build a strong foundation in time series analysis, preparing you for more advanced machine learning topics in the upcoming days. Continue experimenting with the provided code examples, and don't hesitate to explore additional resources to deepen your understanding.

**Happy Learning! 🎉**
