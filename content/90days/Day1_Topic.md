# Day 1: Introduction to Data Science with Python

Welcome to Day 1 of our Data Science series. Today, we'll cover the essentials of Python for Data Science, focusing on the core libraries and techniques used in the industry. This includes an introduction to Python programming, data manipulation with pandas, and data visualization with Matplotlib and Seaborn.

## Table of Contents

1. Introduction to Python
2. Python Basics
    - Data Types
    - Control Structures
3. Introduction to Pandas
    - Series and DataFrames
    - Basic Operations
4. Data Manipulation with Pandas
    - Filtering
    - Aggregation
5. Introduction to Data Visualization
    - Matplotlib Basics
    - Seaborn for Statistical Plots
6. Example Project: Analyzing a Dataset
7. Conclusion and Next Steps

---

## 1. Introduction to Python

Python is a versatile and widely used programming language, especially popular in the field of data science due to its readability, simplicity, and the extensive ecosystem of data-focused libraries.

### Why Python for Data Science?

- **Simplicity & Readability**: Python's syntax is clean and its commands mimic the English language, which makes it an ideal choice for beginners in programming.
- **Extensive Libraries**: Libraries like pandas, NumPy, Scikit-learn, and Matplotlib make data manipulation, statistical analysis, and visualization straightforward.
- **Community & Support**: A vast community of developers and data scientists ensures that Python remains cutting-edge and supportive.

## 2. Python Basics

### Data Types

Python handles a variety of data types. Here's an overview and some examples:

```python
# Integers
x = 5

# Floating point
y = 3.14

# Strings
z = 'Hello Data Science'

# Boolean
a = True
b = False

# None Type
c = None
```

### Control Structures

Control structures in Python direct the flow of your code. Below are examples of `if`, `for`, and `while` loops:

```python
# If statement
if x > 4:
    print('x is greater than 4')

# For loop on a range
for i in range(5):
    print(i)

# While loop
while x > 0:
    print(x)
    x -= 1
```

## 3. Introduction to Pandas

Pandas is a cornerstone library for data manipulation and analysis. It provides data structures and operations for manipulating numerical tables and time series.

### Series and DataFrames

Pandas has two primary data structures:
- **Series**: A one-dimensional array-like object.
- **DataFrame**: A two-dimensional, size-mutable, potentially heterogeneous tabular data.

```python
import pandas as pd

# Creating a Series
s = pd.Series([1, 3, 5, 7, 9])

# Creating a DataFrame
data = {'Product': ['Widget A', 'Widget B'], 'Price': [25.50, 45.75]}
df = pd.DataFrame(data)
```

## 4. Data Manipulation with Pandas

### Filtering

Filtering data is straightforward with pandas:

```python
# Creating a DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})

# Filtering
adults = df[df['age'] >= 30]
print(adults)
```

### Aggregation

Pandas provides several methods to perform aggregations:

```python
# Summing data
total_age = df['age'].sum()
print('Total Age:', total_age)
```

## 5. Introduction to Data Visualization

Visualization is key in data science to understand the data and to communicate findings effectively.

### Matplotlib Basics

Matplotlib is a plotting library for Python which provides a variety of plot types.

```python
import matplotlib.pyplot as plt

# Line plot
plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
plt.show()
```

### Seaborn for Statistical Plots

Seaborn builds on Matplotlib and integrates closely with pandas data structures.

```python
import seaborn as sns

# Load an example dataset
tips = sns.load_dataset('tips')

# Create a bar plot
sns.barplot(x='day', y='total_bill', data=tips)
plt.show()
```

## 6. Example Project: Analyzing a Dataset

Let’s apply what we've learned to a real dataset:

```python
# Load data
data = pd.read_csv('data.csv')

# Display first 5 rows
print(data.head())

# Basic statistics
print(data.describe())

# Visualization
sns.histplot(data['price'])
plt.show()
```

## 7. Conclusion and Next Steps

Congratulations on completing Day 1! You've taken your first steps into Data Science with Python. Continue practicing these concepts and explore more complex analyses and datasets.
