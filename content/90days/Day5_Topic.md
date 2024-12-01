<div style="text-align: center;">
  <h1>🗄️ Day 5: Working with Databases – SQL, PostgreSQL, and Python Integration 🐍💾</h1>
  <p>Efficient Data Storage and Retrieval for Your Data Science Projects!</p>
</div>

---

## 📑 Table of Contents

1. [📅 Review of Day 4](#review-of-day-4)
2. [🗄️ Working with Databases](#working-with-databases-🗄️)
    - [🔍 Introduction to SQL](#introduction-to-sql-🔍)
    - [💾 PostgreSQL Basics](#postgresql-basics-💾)
    - [🐍 Integrating Python with Databases](#integrating-python-with-databases-🐍)
        - [📚 Using SQLAlchemy](#using-sqlalchemy-📚)
        - [📈 Querying with Pandas](#querying-with-pandas-📈)
3. [🛠️📈 Example Project: Database Integration](#example-project-database-integration-🛠️📈)
4. [🚀🎓 Conclusion and Next Steps](#conclusion-and-next-steps-🚀🎓)
5. [📜 Summary of Day 5 📜](#summary-of-day-5-📜)

---

## 1. 📅 Review of Day 4

Before moving forward, let's recap the key concepts we covered on Day 4:

- **Advanced Data Visualization Techniques**: Mastered the use of Plotly for interactive visualizations, explored advanced features of Seaborn, and built interactive dashboards with Plotly Dash.
- **Introduction to Tableau**: Learned how to connect data, build visualizations, and create comprehensive dashboards using Tableau.
- **Example Project: Advanced Data Visualization**: Applied advanced visualization techniques to the Iris dataset, creating interactive and insightful visualizations and dashboards.

With this foundation, we're ready to dive into the world of databases, essential for storing and managing large datasets efficiently.

---

## 2. 🗄️ Working with Databases

Databases are integral to data storage, management, and retrieval. Understanding how to interact with databases is crucial for any data scientist.

### 🔍 Introduction to SQL

**SQL (Structured Query Language)** is the standard language for interacting with relational databases. It allows you to perform various operations such as querying, updating, and managing data.

```sql
-- Creating a new table
CREATE TABLE Employees (
    EmployeeID INT PRIMARY KEY,
    Name VARCHAR(50),
    Age INT,
    Salary DECIMAL(10, 2)
);

-- Inserting data into the table
INSERT INTO Employees (EmployeeID, Name, Age, Salary)
VALUES (1, 'Alice', 30, 70000),
       (2, 'Bob', 25, 50000),
       (3, 'Charlie', 35, 80000);

-- Querying data from the table
SELECT * FROM Employees;

-- Updating data in the table
UPDATE Employees
SET Salary = 75000
WHERE EmployeeID = 1;

-- Deleting data from the table
DELETE FROM Employees
WHERE EmployeeID = 2;
```

**Key Concepts:**

- **CRUD Operations**: Create, Read, Update, Delete.
- **Joins**: Combine rows from two or more tables based on related columns.
- **Aggregations**: Perform calculations on data, such as SUM, AVG, COUNT.

### 💾 PostgreSQL Basics

**PostgreSQL** is a powerful, open-source relational database system. It supports advanced features and is highly extensible.

```sql
-- Connecting to PostgreSQL
\c your_database_name

-- Creating a new database
CREATE DATABASE DataScienceDB;

-- Creating a new schema
CREATE SCHEMA Analysis;

-- Creating a table within the schema
CREATE TABLE Analysis.Sales (
    SaleID SERIAL PRIMARY KEY,
    Product VARCHAR(50),
    Quantity INT,
    Price DECIMAL(10, 2),
    SaleDate DATE
);

-- Inserting data into the table
INSERT INTO Analysis.Sales (Product, Quantity, Price, SaleDate)
VALUES ('Laptop', 5, 1200.00, '2024-01-15'),
       ('Smartphone', 10, 800.00, '2024-01-17'),
       ('Tablet', 7, 500.00, '2024-01-20');

-- Querying data with conditions
SELECT * FROM Analysis.Sales
WHERE Quantity > 5;
```

**Advantages of PostgreSQL:**

- **ACID Compliance**: Ensures reliable transactions.
- **Extensibility**: Supports custom data types, operators, and functions.
- **Community Support**: Extensive documentation and active community.

### 🐍 Integrating Python with Databases

Python offers several libraries to interact with databases, enabling seamless data manipulation and analysis.

#### 📚 Using SQLAlchemy

**SQLAlchemy** is a powerful ORM (Object-Relational Mapping) library that facilitates database interactions in Python.

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the database URL
DATABASE_URL = "postgresql://username:password@localhost:5432/DataScienceDB"

# Create an engine
engine = create_engine(DATABASE_URL)

# Define the base class
Base = declarative_base()

# Define a model
class Employee(Base):
    __tablename__ = 'employees'
    EmployeeID = Column(Integer, primary_key=True)
    Name = Column(String)
    Age = Column(Integer)
    Salary = Column(Float)

# Create the table
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Adding a new employee
new_employee = Employee(Name='David', Age=28, Salary=60000)
session.add(new_employee)
session.commit()

# Querying employees
employees = session.query(Employee).filter(Employee.Age > 25).all()
for emp in employees:
    print(emp.Name, emp.Age, emp.Salary)
```

#### 📈 Querying with Pandas

**Pandas** provides convenient functions to execute SQL queries and load data directly into DataFrames.

```python
import pandas as pd
from sqlalchemy import create_engine

# Define the database URL
DATABASE_URL = "postgresql://username:password@localhost:5432/DataScienceDB"

# Create an engine
engine = create_engine(DATABASE_URL)

# Writing a SQL query
query = "SELECT * FROM Analysis.Sales WHERE Quantity > 5;"

# Loading data into a DataFrame
df_sales = pd.read_sql(query, engine)
print(df_sales.head())

# Performing data analysis
average_price = df_sales['Price'].mean()
print(f"Average Price: {average_price}")
```

**Advantages:**

- **Seamless Integration**: Combine SQL queries with Pandas data manipulation.
- **Efficiency**: Handle large datasets efficiently with optimized database operations.
- **Flexibility**: Perform complex analyses by leveraging the strengths of both SQL and Python.

---

## 3. 🛠️📈 Example Project: Database Integration

Let's apply today's concepts by integrating a PostgreSQL database with Python to manage and analyze sales data.

### 📋 Project Overview

**Objective**: Develop a Python application that connects to a PostgreSQL database, performs CRUD operations, and analyzes sales data.

**Tools**: Python, SQLAlchemy, Pandas, PostgreSQL

### 📝 Step-by-Step Guide

#### 1. Set Up PostgreSQL Database

```sql
-- Creating the Sales table
CREATE TABLE Analysis.Sales (
    SaleID SERIAL PRIMARY KEY,
    Product VARCHAR(50),
    Quantity INT,
    Price DECIMAL(10, 2),
    SaleDate DATE
);

-- Inserting sample data
INSERT INTO Analysis.Sales (Product, Quantity, Price, SaleDate)
VALUES 
('Laptop', 5, 1200.00, '2024-01-15'),
('Smartphone', 10, 800.00, '2024-01-17'),
('Tablet', 7, 500.00, '2024-01-20');
```

#### 2. Connect Python to PostgreSQL Using SQLAlchemy

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database credentials
DATABASE_URL = "postgresql://username:password@localhost:5432/DataScienceDB"

# Create engine
engine = create_engine(DATABASE_URL)

# Define base
Base = declarative_base()

# Define Sales model
class Sales(Base):
    __tablename__ = 'Sales'
    SaleID = Column(Integer, primary_key=True)
    Product = Column(String)
    Quantity = Column(Integer)
    Price = Column(Float)
    SaleDate = Column(Date)

# Create table
Base.metadata.create_all(engine)

# Create session
Session = sessionmaker(bind=engine)
session = Session()
```

#### 3. Perform CRUD Operations

```python
# Create a new sale
new_sale = Sales(Product='Monitor', Quantity=3, Price=300.00, SaleDate='2024-01-22')
session.add(new_sale)
session.commit()

# Read sales data
sales = session.query(Sales).all()
for sale in sales:
    print(sale.Product, sale.Quantity, sale.Price, sale.SaleDate)

# Update a sale
sale_to_update = session.query(Sales).filter(Sales.Product == 'Tablet').first()
sale_to_update.Price = 550.00
session.commit()

# Delete a sale
sale_to_delete = session.query(Sales).filter(Sales.Product == 'Laptop').first()
session.delete(sale_to_delete)
session.commit()
```

#### 4. Analyze Sales Data with Pandas

```python
import pandas as pd

# Create an engine
engine = create_engine(DATABASE_URL)

# Query data into DataFrame
df_sales = pd.read_sql("SELECT * FROM Analysis.Sales;", engine)
print(df_sales.head())

# Calculate total revenue
df_sales['Revenue'] = df_sales['Quantity'] * df_sales['Price']
total_revenue = df_sales['Revenue'].sum()
print(f"Total Revenue: {total_revenue}")

# Group by Product
product_sales = df_sales.groupby('Product')['Revenue'].sum().reset_index()
print(product_sales)
```

---

## 4. 🚀🎓 Conclusion and Next Steps

Congratulations on completing **Day 5**! Today, you learned how to interact with databases using SQL and PostgreSQL, integrated Python with databases using SQLAlchemy and Pandas, and developed a project that combines these skills to manage and analyze sales data effectively.

### 🔮 What’s Next?

- **Day 6: Deep Learning Basics**: Introduction to neural networks, TensorFlow, and Keras for building deep learning models.
- **Day 7: Natural Language Processing (NLP)**: Explore techniques for processing and analyzing textual data.
- **Day 8: Big Data Tools**: Introduction to Hadoop, Spark, and other big data technologies.
- **Ongoing Projects**: Continue developing projects to apply your skills in real-world scenarios, enhancing both your portfolio and practical understanding.

### 📝 Tips for Success

- **Practice Regularly**: Consistently apply what you've learned through exercises and projects to reinforce your knowledge.
- **Engage with the Community**: Participate in forums, attend webinars, and collaborate with peers to broaden your perspective and solve challenges together.
- **Stay Curious**: Continuously explore new libraries, tools, and methodologies to stay ahead in the ever-evolving field of data science.
- **Document Your Work**: Keep detailed notes and document your projects to track your progress and facilitate future learning.

Keep up the outstanding work, and stay motivated as you continue your Data Science journey! 🚀📚

---

<div style="text-align: center;">
  <p>✨ Keep Learning, Keep Growing! ✨</p>
  <p>🚀 Your Data Science Journey Continues 🚀</p>
  <p>📚 Happy Coding! 🎉</p>
</div>

---

# 📜 Summary of Day 5 📜

- **🗄️ Working with Databases**: Learned how to use SQL and PostgreSQL for data storage and management.
- **🔍 Introduction to SQL**: Mastered CRUD operations and key SQL concepts.
- **💾 PostgreSQL Basics**: Set up and interacted with a PostgreSQL database.
- **🐍 Integrating Python with Databases**: Utilized SQLAlchemy and Pandas to connect Python applications with databases.
- **🛠️📈 Example Project: Database Integration**: Developed a Python application to manage and analyze sales data using PostgreSQL.

This structured approach ensures that you build a robust foundation in database management and integration, essential for handling large-scale data in your data science projects. Continue experimenting with the provided tools and don't hesitate to delve into additional resources to deepen your expertise.

**Happy Learning! 🎉**
