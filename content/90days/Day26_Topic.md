<div style="text-align: center;">
  <h1 style="color:#FF5722;">🚀 Becoming a Scikit-Learn Boss in 90 Days: Day 26 – Deploying Models with FastAPI and Docker 🛠️🌐</h1>
  <p style="font-size:18px;">Enhance Your Model Deployment Skills with FastAPI and Docker for Scalable, Efficient Applications!</p>
  
  <!-- Animated Header Image -->
  <img src="https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif" alt="FastAPI and Docker Animation" width="600">
</div>

---

## 📑 Table of Contents

1. [🌟 Welcome to Day 26](#welcome-to-day-26)
2. [🔍 Review of Day 25 📜](#review-of-day-25-📜)
3. [🧠 Introduction to FastAPI and Docker 🧠](#introduction-to-fastapi-and-docker-🧠)
    - [📚 What is FastAPI?](#what-is-fastapi-📚)
    - [📚 What is Docker?](#what-is-docker-📚)
    - [🔍 Benefits of Using FastAPI and Docker](#benefits-of-using-fastapi-and-docker-🔍)
4. [🛠️ Setting Up FastAPI for Model Deployment 🛠️](#setting-up-fastapi-for-model-deployment-🛠️)
    - [📊 Installing FastAPI and Uvicorn](#installing-fastapi-and-uvicorn-📊)
    - [📊 Building a Simple FastAPI Application](#building-a-simple-fastapi-application-📊)
    - [📊 Creating API Endpoints for Predictions](#creating-api-endpoints-for-predictions-📊)
5. [🛠️ Containerizing Your Application with Docker 🛠️](#containerizing-your-application-with-docker-🛠️)
    - [📊 Writing a Dockerfile for FastAPI](#writing-a-dockerfile-for-fastapi-📊)
    - [📊 Building and Running the Docker Container](#building-and-running-the-docker-container-📊)
    - [📊 Managing Dependencies with Docker](#managing-dependencies-with-docker-📊)
6. [📈 Example Project: Deploying a Scikit-Learn Model with FastAPI and Docker 📈](#example-project-deploying-a-scikit-learn-model-with-fastapi-and-docker-📈)
    - [📋 Project Overview](#project-overview-📋)
    - [📝 Step-by-Step Guide](#step-by-step-guide-📝)
        - [1. Train and Save the Scikit-Learn Model](#1-train-and-save-the-scikit-learn-model)
        - [2. Set Up the FastAPI Application](#2-set-up-the-fastapi-application)
        - [3. Create API Endpoints for Prediction](#3-create-api-endpoints-for-prediction)
        - [4. Dockerize the FastAPI Application](#4-dockerize-the-fastapi-application)
        - [5. Deploy the Docker Container](#5-deploy-the-docker-container)
        - [6. Test the Deployed API](#6-test-the-deployed-api)
    - [📊 Results and Insights](#results-and-insights-📊)
7. [🚀🎓 Conclusion and Next Steps 🚀🎓](#conclusion-and-next-steps-🚀🎓)
8. [📜 Summary of Day 26 📜](#summary-of-day-26-📜)

---

## 1. 🌟 Welcome to Day 26

Welcome to **Day 26** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, we'll focus on **Deploying Machine Learning Models with FastAPI and Docker**. FastAPI is a modern, fast (high-performance) web framework for building APIs with Python, while Docker enables you to containerize your applications for consistent and scalable deployments. By mastering these tools, you'll be able to serve your Scikit-Learn models efficiently, ensuring they are accessible and reliable in production environments.

<!-- Animated Divider -->
<img src="https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif" alt="Divider Animation" width="100%">

---

## 2. 🔍 Review of Day 25 📜

Before diving into today's topic, let's briefly recap what we covered yesterday:

- **Deploying Machine Learning Models to Production**: Learned the importance of model deployment, challenges involved, and various deployment strategies to transform models into real-world applications.
- **Deployment Techniques and Tools**: Explored key tools and frameworks such as Flask, FastAPI, Docker, and cloud platforms for effective model deployment.
- **Implementing Model Deployment with Flask**: Built a Flask API to serve a Scikit-Learn model, containerized the application using Docker, and deployed it to Heroku.
- **Example Project**: Successfully deployed a trained Iris classification model as a RESTful API, demonstrating practical application of deployment techniques.

With a solid understanding of deploying models using Flask and Docker, we're now ready to explore a more modern and efficient framework—FastAPI—for model deployment.

---

## 3. 🧠 Introduction to FastAPI and Docker 🧠

### 📚 What is FastAPI?

**FastAPI** is a modern, high-performance web framework for building APIs with Python 3.6+ based on standard Python type hints. It offers automatic interactive API documentation, data validation, and asynchronous support, making it an excellent choice for deploying machine learning models.

**Key Features:**
- **Fast**: High performance, on par with NodeJS and Go.
- **Easy to Use**: Minimal boilerplate code, intuitive design.
- **Automatic Documentation**: Swagger UI and ReDoc available out-of-the-box.
- **Type Safety**: Leverages Python type hints for data validation and error checking.

### 📚 What is Docker?

**Docker** is a platform that allows you to automate the deployment, scaling, and management of applications within lightweight, portable containers. Containers encapsulate an application along with its dependencies, ensuring consistency across different environments.

**Key Benefits:**
- **Portability**: Run containers anywhere—local machines, servers, cloud platforms.
- **Isolation**: Each container operates independently, preventing conflicts.
- **Scalability**: Easily scale applications by deploying multiple container instances.
- **Efficiency**: Lightweight compared to traditional virtual machines, with faster startup times.

### 🔍 Benefits of Using FastAPI and Docker

- **Consistency**: Docker ensures that your FastAPI application runs the same way in development, testing, and production.
- **Scalability**: Easily scale your API by deploying multiple Docker containers behind a load balancer.
- **Maintainability**: Simplify dependency management and updates with containerization.
- **Security**: Isolate your application from the host system, enhancing security.
- **Efficiency**: FastAPI's high performance combined with Docker's lightweight containers leads to efficient resource utilization.

---

## 4. 🛠️ Setting Up FastAPI for Model Deployment 🛠️

### 📊 Installing FastAPI and Uvicorn

First, install FastAPI and Uvicorn, an ASGI server for running FastAPI applications.

```bash
pip install fastapi uvicorn
```

### 📊 Building a Simple FastAPI Application

Create a file named `main.py` with the following content to set up a basic FastAPI application.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Define the request model
class InputData(BaseModel):
    input: list

# Load the trained Scikit-Learn model
model = joblib.load('model.pkl')

@app.post("/predict")
def predict(data: InputData):
    input_features = np.array(data.input).reshape(1, -1)
    prediction = model.predict(input_features)
    return {"prediction": prediction.tolist()}
```

### 📊 Creating API Endpoints for Predictions

The `/predict` endpoint accepts POST requests with JSON data containing the input features and returns the model's prediction.

**Example Request:**

```json
{
  "input": [5.1, 3.5, 1.4, 0.2]
}
```

**Example Response:**

```json
{
  "prediction": [0]
}
```

You can run the FastAPI application using Uvicorn:

```bash
uvicorn main:app --reload
```

Access the interactive API documentation at `http://127.0.0.1:8000/docs`.

---

## 5. 🛠️ Containerizing Your Application with Docker 🛠️

### 📊 Writing a Dockerfile for FastAPI

Create a `Dockerfile` to containerize your FastAPI application.

```dockerfile
# Use the official Python image as the base
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port FastAPI is running on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 📊 Building and Running the Docker Container

1. **Create a `requirements.txt` File:**

   ```txt
   fastapi
   uvicorn
   scikit-learn
   joblib
   numpy
   ```

2. **Build the Docker Image:**

   ```bash
   docker build -t scikit-learn-fastapi .
   ```

3. **Run the Docker Container:**

   ```bash
   docker run -d -p 8000:8000 scikit-learn-fastapi
   ```

4. **Access the API:**

   Open your browser and navigate to `http://localhost:8000/docs` to access the interactive API documentation.

### 📊 Managing Dependencies with Docker

Docker ensures that all dependencies are bundled within the container, eliminating discrepancies between development and production environments. Updating dependencies is as simple as modifying the `requirements.txt` and rebuilding the Docker image.

---

## 6. 📈 Example Project: Deploying a Scikit-Learn Model with FastAPI and Docker 📈

### 📋 Project Overview

**Objective**: Deploy a trained Scikit-Learn Iris classification model as a RESTful API using FastAPI and Docker. This project involves training the model, creating a FastAPI application, containerizing it with Docker, and deploying it to a cloud platform like Heroku.

**Tools**: Python, FastAPI, Scikit-Learn, Joblib, Docker, Heroku, Postman

### 📝 Step-by-Step Guide

#### 1. Train and Save the Scikit-Learn Model

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')
```

#### 2. Set Up the FastAPI Application

Create a file named `main.py` with the following content:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Define the request model
class InputData(BaseModel):
    input: list

# Load the trained Scikit-Learn model
model = joblib.load('model.pkl')

@app.post("/predict")
def predict(data: InputData):
    input_features = np.array(data.input).reshape(1, -1)
    prediction = model.predict(input_features)
    return {"prediction": prediction.tolist()}
```

#### 3. Create API Endpoints for Prediction

The `/predict` endpoint accepts POST requests with input features and returns the model's prediction.

#### 4. Dockerize the FastAPI Application

1. **Create a `requirements.txt` File:**

   ```txt
   fastapi
   uvicorn
   scikit-learn
   joblib
   numpy
   ```

2. **Create a `Dockerfile`:**

   ```dockerfile
   # Use the official Python image as the base
   FROM python:3.8-slim

   # Set environment variables
   ENV PYTHONDONTWRITEBYTECODE=1
   ENV PYTHONUNBUFFERED=1

   # Set the working directory
   WORKDIR /app

   # Copy the requirements file
   COPY requirements.txt .

   # Install dependencies
   RUN pip install --upgrade pip
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy the application code
   COPY . .

   # Expose the port FastAPI is running on
   EXPOSE 8000

   # Command to run the application
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

3. **Build and Run the Docker Container:**

   ```bash
   # Build the Docker image
   docker build -t iris-fastapi-app .

   # Run the Docker container
   docker run -d -p 8000:8000 iris-fastapi-app
   ```

4. **Access the API Documentation:**

   Navigate to `http://localhost:8000/docs` to interact with the API using Swagger UI.

#### 5. Deploy to a Cloud Platform (e.g., Heroku)

**Step 1**: Install the Heroku CLI and log in.

```bash
# Install Heroku CLI (if not already installed)
# Follow instructions at https://devcenter.heroku.com/articles/heroku-cli

# Log in to Heroku
heroku login
```

**Step 2**: Create a `Procfile` for Heroku.

```txt
web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-5000}
```

**Step 3**: Initialize a Git repository, commit your code, and deploy.

```bash
# Initialize Git repository
git init
git add .
git commit -m "Initial commit"

# Create a Heroku app
heroku create iris-fastapi-app

# Push the code to Heroku
git push heroku master

# Scale the dynos
heroku ps:scale web=1
```

#### 6. Test the Deployed API

Use Postman or cURL to send a POST request to your Heroku app.

```bash
curl -X POST https://iris-fastapi-app.herokuapp.com/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [5.1, 3.5, 1.4, 0.2]}'
```

**Expected Response:**

```json
{
  "prediction": [0]
}
```

#### 7. Visualize the Results

You should receive a JSON response with the prediction, indicating the Iris species based on the input features.

---

## 7. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 26** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, you mastered the essentials of **Deploying Machine Learning Models with FastAPI and Docker**, learning how to build robust APIs for your Scikit-Learn models and containerize them for scalable deployments. By implementing a FastAPI application and containerizing it with Docker, you ensured that your models are accessible, consistent, and ready for production environments.

### 🔮 What’s Next?

- **Days 27-30: Advanced Deployment Techniques**
  - **Day 27**: Scaling Machine Learning Models with Kubernetes
  - **Day 28**: Monitoring and Maintaining Deployed Models
  - **Day 29**: Deploying Models to Multiple Cloud Platforms (AWS, GCP, Azure)
  - **Day 30**: Securing Machine Learning APIs

- **Days 31-90: Specialized Topics and Comprehensive Projects**
  - Explore areas like advanced ensemble methods, model optimization, and deploying models to cloud platforms.
  - Engage in larger projects that integrate multiple machine learning techniques to solve complex real-world problems.

### 📝 Tips for Success

- **Practice Regularly**: Continuously apply deployment techniques through projects to reinforce your understanding.
- **Engage with the Community**: Participate in forums, attend webinars, and collaborate with peers to exchange knowledge and tackle challenges together.
- **Stay Curious**: Keep exploring new deployment tools, frameworks, and best practices to stay ahead in the field.
- **Document Your Work**: Maintain a detailed journal or portfolio of your deployment projects to track progress and showcase your skills to potential employers or collaborators.

Keep up the excellent work, and stay motivated as you continue your journey to mastering Scikit-Learn and becoming a proficient machine learning practitioner! 🚀📚

---

<div style="text-align: center;">
  <!-- Animated Footer Image -->
  <img src="https://media.giphy.com/media/l0HlBO7eyXzSZkJri/giphy.gif" alt="Happy Coding" width="300">
</div>

---

# 📜 Summary of Day 26 📜

- **🧠 Introduction to FastAPI and Docker**: Gained a comprehensive understanding of FastAPI, a modern web framework for building APIs, and Docker, a platform for containerizing applications to ensure consistency across environments.
- **📊 Setting Up FastAPI for Model Deployment**: Learned how to install FastAPI and Uvicorn, build a simple FastAPI application, and create API endpoints for serving Scikit-Learn model predictions.
- **🛠️ Containerizing Your Application with Docker**: Explored how to write a Dockerfile for a FastAPI application, build and run Docker containers, and manage dependencies effectively.
- **📈 Example Project: Deploying a Scikit-Learn Model with FastAPI and Docker**: Successfully deployed a trained Iris classification model as a RESTful API using FastAPI and Docker, including steps for training the model, setting up the API, containerizing the application, and deploying it to Heroku.
- **🛠️📈 Practical Skills Acquired**: Enhanced ability to serve machine learning models via FastAPI APIs, containerize applications with Docker for consistency and scalability, and deploy models to cloud platforms for real-world accessibility.
