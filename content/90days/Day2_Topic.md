# Day 2: Data Cleaning, Advanced Pandas, NumPy, and Exploratory Data Analysis 📊🧹🔍

Welcome to Day 2 of our Data Science series! Today, we'll delve deeper into the essential steps of data science: data cleaning and preprocessing, advanced data manipulation with Pandas, numerical computations with NumPy, and Exploratory Data Analysis (EDA). We'll also introduce the basics of machine learning using scikit-learn. By the end of this day, you'll have a solid understanding of how to prepare and analyze data effectively.

## Table of Contents 📚

1. [Review of Day 1](#review-of-day-1)
2. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
    - Handling Missing Values
    - Removing Duplicates
    - Data Transformation
3. [Advanced Pandas Techniques](#advanced-pandas-techniques)
    - Merging and Joining DataFrames
    - Grouping and Aggregation
    - Pivot Tables
4. [Introduction to NumPy](#introduction-to-numpy)
    - NumPy Arrays
    - Array Operations
    - Broadcasting
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - Descriptive Statistics
    - Data Visualization for EDA
6. [Introduction to Machine Learning with scikit-learn](#introduction-to-machine-learning-with-scikit-learn)
    - Supervised vs. Unsupervised Learning
    - Simple Linear Regression Example
7. [Example Project: Data Cleaning and EDA](#example-project-data-cleaning-and-eda)
8. [Conclusion and Next Steps](#conclusion-and-next-steps)

---

## 1. Review of Day 1 📅

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Introduction to Python**: Understanding why Python is a preferred language for data science.
- **Python Basics**: Learning about data types and control structures (`if`, `for`, `while`).
- **Introduction to Pandas**: Exploring Series and DataFrames for data manipulation.
- **Data Manipulation with Pandas**: Performing filtering and aggregation.
- **Introduction to Data Visualization**: Creating basic plots with Matplotlib and Seaborn.
- **Example Project**: Analyzing a simple dataset to apply learned concepts.

---

## 2. Data Cleaning and Preprocessing 🧹🔧

Data cleaning and preprocessing are critical steps in the data science workflow. Real-world data is often messy and requires meticulous preparation before analysis.

### Handling Missing Values ❓

Missing values can skew your analysis and lead to inaccurate results. Pandas provides several methods to handle them:

```python
import pandas as pd

# Creating a DataFrame with missing values
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, None, 35, 40],
    'Salary': [50000, 60000, None, 80000]
}
df = pd.DataFrame(data)

# Detecting missing values
print(df.isnull())

# Dropping rows with any missing values
df_cleaned = df.dropna()
print(df_cleaned)

# Filling missing values with a specific value
df_filled = df.fillna({'Age': df['Age'].mean(), 'Salary': df['Salary'].median()})
print(df_filled)
```

### Removing Duplicates 🗑️

Duplicate records can distort analysis. Use Pandas to identify and remove them:

```python
# Creating a DataFrame with duplicate rows
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Bob'],
    'Age': [25, 30, 35, 30],
    'Salary': [50000, 60000, 70000, 60000]
}
df = pd.DataFrame(data)

# Detecting duplicates
print(df.duplicated())

# Removing duplicates
df_unique = df.drop_duplicates()
print(df_unique)
```

### Data Transformation 🔄

Transforming data into a suitable format is often necessary for analysis:

- **Normalization**: Scaling numerical data to a standard range.
- **Encoding Categorical Variables**: Converting categories into numerical codes.

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Normalizing numerical data
scaler = StandardScaler()
df['Salary_Normalized'] = scaler.fit_transform(df[['Salary']])
print(df)

# Encoding categorical data
le = LabelEncoder()
df['Name_Encoded'] = le.fit_transform(df['Name'])
print(df)
```

---

## 3. Advanced Pandas Techniques 📈📉

Building upon the basics, let's explore some advanced Pandas functionalities that enhance data manipulation capabilities.

### Merging and Joining DataFrames 🔗

Combining multiple DataFrames is a common task in data analysis.

```python
# Creating two DataFrames
df1 = pd.DataFrame({
    'EmployeeID': [1, 2, 3],
    'Name': ['Alice', 'Bob', 'Charlie']
})

df2 = pd.DataFrame({
    'EmployeeID': [1, 2, 4],
    'Salary': [50000, 60000, 70000]
})

# Merging DataFrames on EmployeeID
merged_df = pd.merge(df1, df2, on='EmployeeID', how='inner')  # Inner join
print(merged_df)

# Outer join
outer_df = pd.merge(df1, df2, on='EmployeeID', how='outer')
print(outer_df)
```

### Grouping and Aggregation 📊

Grouping data allows you to perform aggregate operations on subsets of data.

```python
# Sample DataFrame
data = {
    'Department': ['Sales', 'Sales', 'HR', 'HR', 'IT'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Salary': [50000, 60000, 55000, 65000, 70000]
}
df = pd.DataFrame(data)

# Grouping by Department and calculating mean salary
grouped = df.groupby('Department')['Salary'].mean()
print(grouped)

# Multiple aggregations
aggregated = df.groupby('Department').agg({'Salary': ['mean', 'max', 'min']})
print(aggregated)
```

### Pivot Tables 📑

Pivot tables summarize data, making it easier to analyze patterns.

```python
# Creating a pivot table
pivot = pd.pivot_table(df, values='Salary', index='Department', columns='Employee', aggfunc='mean')
print(pivot)
```

---

## 4. Introduction to NumPy 🧮🔢

NumPy is a fundamental library for numerical computations in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on them.

### NumPy Arrays 🌀

Arrays are similar to lists but offer more functionality and efficiency for numerical operations.

```python
import numpy as np

# Creating a NumPy array
arr = np.array([1, 2, 3, 4, 5])
print(arr)

# Multi-dimensional arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix)
```

### Array Operations ➕➖✖️➗

NumPy allows element-wise operations on arrays.

```python
# Arithmetic operations
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

print(a + b)  # Addition
print(a - b)  # Subtraction
print(a * b)  # Multiplication
print(a / b)  # Division
```

### Broadcasting 📡

Broadcasting enables arithmetic operations between arrays of different shapes.

```python
# Broadcasting example
a = np.array([1, 2, 3])
b = 2
print(a + b)  # Adds 2 to each element of a

# Broadcasting with multi-dimensional arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])
print(matrix + vector)
```

---

## 5. Exploratory Data Analysis (EDA) 🔍📈

EDA involves analyzing datasets to summarize their main characteristics, often using visual methods. It helps in understanding the data distribution, detecting outliers, and identifying patterns.

### Descriptive Statistics 📊

Descriptive statistics provide simple summaries about the data.

```python
# Sample DataFrame
data = {
    'Age': [25, 30, 35, 40, 45],
    'Salary': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)

# Descriptive statistics
print(df.describe())
```

### Data Visualization for EDA 🎨

Visualizations help in uncovering insights from the data.

- **Histograms**: Show the distribution of a single variable.
- **Box Plots**: Highlight the distribution and detect outliers.
- **Scatter Plots**: Examine the relationship between two variables.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.figure(figsize=(8, 4))
sns.histplot(df['Age'], bins=5, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box Plot
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Salary'])
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Salary', data=df)
plt.title('Age vs. Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
```

---

## 6. Introduction to Machine Learning with scikit-learn 🤖📚

Machine learning (ML) is a subset of artificial intelligence that focuses on building systems that learn from data to make predictions or decisions. scikit-learn is a powerful Python library for ML.

### Supervised vs. Unsupervised Learning 🏷️🔍

- **Supervised Learning**: The model is trained on labeled data. Examples include classification and regression.
- **Unsupervised Learning**: The model works on unlabeled data to find hidden patterns. Examples include clustering and association.

### Simple Linear Regression Example 📈

Linear regression predicts a target variable based on one or more predictor variables.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample DataFrame
data = {
    'Experience': [1, 2, 3, 4, 5],
    'Salary': [50000, 60000, 70000, 80000, 90000]
}
df = pd.DataFrame(data)

# Features and target
X = df[['Experience']]
y = df['Salary']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Plotting the regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Experience', y='Salary', data=df, label='Actual')
sns.lineplot(x=X_test['Experience'], y=y_pred, color='red', label='Predicted')
plt.title('Experience vs. Salary')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

---

## 7. Example Project: Data Cleaning and EDA 🛠️📈

Let's apply the concepts learned today to a real-world dataset. We'll perform data cleaning, preprocessing, and exploratory data analysis.

### Project Overview

**Objective**: Analyze a dataset to uncover insights by cleaning the data and performing EDA.

**Dataset**: [Titanic Dataset](https://www.kaggle.com/c/titanic/data) *(Ensure you have the dataset downloaded and placed in your working directory)*

### Step-by-Step Guide

1. **Load the Dataset**

    ```python
    import pandas as pd

    # Loading the dataset
    df = pd.read_csv('titanic.csv')
    print(df.head())
    ```

2. **Understanding the Data**

    ```python
    # Data information
    print(df.info())

    # Descriptive statistics
    print(df.describe())
    ```

3. **Handling Missing Values**

    ```python
    # Checking for missing values
    print(df.isnull().sum())

    # Dropping columns with excessive missing values
    df = df.drop(['Cabin', 'Ticket'], axis=1)

    # Filling missing values in 'Age' with median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Filling missing values in 'Embarked' with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    ```

4. **Data Transformation**

    ```python
    from sklearn.preprocessing import LabelEncoder

    # Encoding categorical variables
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    ```

5. **Exploratory Data Analysis**

    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Survival rate
    sns.countplot(x='Survived', data=df)
    plt.title('Survival Count')
    plt.show()

    # Age distribution by survival
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', kde=True)
    plt.title('Age Distribution by Survival')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    ```

6. **Summary of Insights**

    - **Survival Rate**: Analyze the number of survivors vs. non-survivors.
    - **Age Impact**: Determine if age influenced survival chances.
    - **Sex and Survival**: Explore the relationship between gender and survival.
    - **Embarked Port**: Investigate if the port of embarkation had any effect on survival.

---

## 8. Conclusion and Next Steps 🚀🎓

Congratulations on completing Day 2! Today, you mastered the crucial skills of data cleaning and preprocessing, explored advanced data manipulation with Pandas, got introduced to NumPy for numerical computations, and conducted Exploratory Data Analysis to uncover valuable insights from data. Additionally, you took your first steps into the realm of machine learning with a simple linear regression example.

### What’s Next?

- **Day 3: Advanced Machine Learning Techniques**: Dive deeper into machine learning algorithms, including classification, clustering, and model evaluation.
- **Day 4: Data Visualization Mastery**: Learn advanced visualization techniques to create insightful and interactive plots.
- **Day 5: Working with Databases**: Understand how to interact with SQL databases and perform data extraction.
- **Ongoing Projects**: Start working on personal or guided projects to apply your skills in real-world scenarios.

### Tips for Success 📝

- **Practice Regularly**: Consistently apply what you've learned through exercises and projects.
- **Engage with the Community**: Join forums, attend webinars, and collaborate with peers to enhance your learning experience.
- **Stay Curious**: Explore new libraries, tools, and techniques to stay updated in the ever-evolving field of data science.

Keep up the great work, and see you tomorrow for more exciting lessons!

---

# Summary of Day 2 📜

- **Data Cleaning and Preprocessing**: Essential for ensuring data quality and reliability.
- **Advanced Pandas Techniques**: Enhances data manipulation capabilities, making complex operations easier.
- **Introduction to NumPy**: Fundamental for numerical computations, enabling efficient data processing.
- **Exploratory Data Analysis (EDA)**: Critical for understanding data distributions and uncovering patterns.
- **Introduction to Machine Learning**: Provides a glimpse into predictive modeling and the basics of scikit-learn.

This structured approach ensures that you build a strong foundation, preparing you for more advanced topics in the upcoming days. Keep experimenting with the code examples, and don't hesitate to explore additional resources to deepen your understanding.

Happy Learning! 🎉