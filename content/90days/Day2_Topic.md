<div style="text-align: center;">
  <h1>🚀 Becoming a Scikit-Learn Boss in 90 Days: Day 2 – Supervised Learning: Classification Algorithms 🐍📊</h1>
  <p>Dive Deeper into Classification Techniques to Enhance Your Machine Learning Models!</p>
</div>

---

## 📑 Table of Contents

1. [🌟 Welcome to Day 2](#welcome-to-day-2)
2. [🔍 Review of Day 1 📜](#review-of-day-1-📜)
3. [🧠 Introduction to Supervised Learning: Classification](#introduction-to-supervised-learning-classification-🧠)
    - [📚 What is Classification?](#what-is-classification-📚)
    - [🔍 Types of Classification Problems](#types-of-classification-problems-🔍)
4. [📊 Classification Algorithms](#classification-algorithms-📊)
    - [🟢 Logistic Regression](#logistic-regression-🟢)
    - [🌳 Decision Trees](#decision-trees-🌳)
    - [👫 K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn-👫)
    - [🔧 Support Vector Machines (SVM)](#support-vector-machines-svm-🔧)
5. [🛠️ Implementing Classification Algorithms with Scikit-Learn](#implementing-classification-algorithms-with-scikit-learn-🛠️)
    - [🟢 Logistic Regression Example](#logistic-regression-example-🟢)
    - [🌳 Decision Tree Example](#decision-tree-example-🌳)
    - [👫 K-Nearest Neighbors Example](#k-nearest-neighbors-example-👫)
    - [🔧 Support Vector Machines Example](#support-vector-machines-example-🔧)
6. [📈 Model Evaluation for Classification](#model-evaluation-for-classification-📈)
    - [✅ Accuracy](#accuracy-✅)
    - [📏 Precision, Recall, and F1-Score](#precision-recall-and-f1-score-📏)
    - [🔍 Confusion Matrix](#confusion-matrix-🔍)
    - [📈 ROC Curve and AUC](#roc-curve-and-auc-📈)
7. [🛠️📈 Example Project: Advanced Iris Classification](#example-project-advanced-iris-classification-🛠️📈)
8. [🚀🎓 Conclusion and Next Steps](#conclusion-and-next-steps-🚀🎓)
9. [📜 Summary of Day 2 📜](#summary-of-day-2-📜)

---

## 1. 🌟 Welcome to Day 2

Welcome to **Day 2** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, we'll delve into **Supervised Learning**, focusing specifically on **Classification Algorithms**. You'll learn about different classification techniques, implement them using Scikit-Learn, and evaluate their performance to build more accurate and reliable models.

---

## 2. 🔍 Review of Day 1 📜

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Introduction to Scikit-Learn**: Understanding its role in machine learning.
- **Setting Up Your Environment**: Installed Scikit-Learn and set up a virtual environment.
- **Understanding Scikit-Learn's API**: Explored the Estimator API, fit and predict methods, and pipelines.
- **Basic Data Preprocessing**: Handled missing values, encoded categorical variables, and scaled features.
- **Building Your First Model**: Developed a simple Logistic Regression classifier using the Iris dataset.
- **Model Evaluation Metrics**: Evaluated the model using accuracy, precision, recall, F1-score, and confusion matrix.
- **Example Project: Iris Classification**: Completed a full machine learning pipeline from data loading to model evaluation.

With this foundation, we're ready to explore various classification algorithms that will enhance your machine learning toolkit.

---

## 3. 🧠 Introduction to Supervised Learning: Classification 🧠

### 📚 What is Classification?

**Classification** is a type of supervised learning where the goal is to predict the categorical label of new observations based on past observations with known labels.

### 🔍 Types of Classification Problems

- **Binary Classification**: Two possible classes (e.g., spam vs. not spam).
- **Multiclass Classification**: More than two classes (e.g., species classification).
- **Multilabel Classification**: Multiple labels can be assigned to each observation.

---

## 4. 📊 Classification Algorithms 📊

### 🟢 Logistic Regression 🟢

A statistical method for binary classification that models the probability of a binary outcome.

### 🌳 Decision Trees 🌳

A non-parametric model that splits data into subsets based on feature values, creating a tree-like structure for decision making.

### 👫 K-Nearest Neighbors (KNN) 👫

A simple, instance-based learning algorithm that classifies new instances based on the majority class among their K nearest neighbors.

### 🔧 Support Vector Machines (SVM) 🔧

A powerful classifier that finds the optimal hyperplane separating different classes by maximizing the margin between them.

---

## 5. 🛠️ Implementing Classification Algorithms with Scikit-Learn 🛠️

### 🟢 Logistic Regression Example 🟢

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Initialize the model
log_reg = LogisticRegression(max_iter=200)

# Train the model
log_reg.fit(X_train_scaled, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Evaluate the model
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))
```

### 🌳 Decision Tree Example 🌳

```python
from sklearn.tree import DecisionTreeClassifier

# Initialize the model
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the model
decision_tree.fit(X_train_scaled, y_train)

# Make predictions
y_pred_tree = decision_tree.predict(X_test_scaled)

# Evaluate the model
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
```

### 👫 K-Nearest Neighbors Example 👫

```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize the model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train_scaled, y_train)

# Make predictions
y_pred_knn = knn.predict(X_test_scaled)

# Evaluate the model
print("K-Nearest Neighbors Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
```

### 🔧 Support Vector Machines Example 🔧

```python
from sklearn.svm import SVC

# Initialize the model
svm = SVC(kernel='linear', probability=True, random_state=42)

# Train the model
svm.fit(X_train_scaled, y_train)

# Make predictions
y_pred_svm = svm.predict(X_test_scaled)

# Evaluate the model
print("Support Vector Machines Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
```

---

## 6. 📈 Model Evaluation for Classification 📈

### ✅ Accuracy ✅

Measures the proportion of correct predictions out of all predictions made.

### 📏 Precision, Recall, and F1-Score 📏

- **Precision**: The ratio of true positive predictions to the total predicted positives.
- **Recall**: The ratio of true positive predictions to the actual positives.
- **F1-Score**: The harmonic mean of precision and recall.

### 🔍 Confusion Matrix 🔍

A table used to describe the performance of a classification model by comparing actual vs. predicted labels.

### 📈 ROC Curve and AUC 📈

- **ROC Curve**: A graphical plot illustrating the diagnostic ability of a binary classifier.
- **AUC (Area Under the Curve)**: Measures the entire two-dimensional area underneath the ROC curve, providing an aggregate measure of performance.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarize the output for ROC
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_binarized.shape[1]

# Predict probabilities
y_score = svm.predict_proba(X_test_scaled)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure()
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

---

## 7. 🛠️📈 Example Project: Advanced Iris Classification 🛠️📈

Let's consolidate what you've learned by building an advanced classification pipeline using the Iris dataset.

### 📋 Project Overview

**Objective**: Develop a comprehensive machine learning pipeline to classify Iris species, incorporating multiple classification algorithms and evaluating their performance.

**Tools**: Python, Scikit-Learn, pandas, Matplotlib, Seaborn

### 📝 Step-by-Step Guide

#### 1. Load and Explore the Dataset

```python
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='Species')

# Combine features and target
df = pd.concat([X, y], axis=1)
print(df.head())

# Visualize pairplot
sns.pairplot(df, hue='Species', palette='Set1')
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Initialize models
log_reg = LogisticRegression(max_iter=200)
decision_tree = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', probability=True, random_state=42)

# Train models
log_reg.fit(X_train_scaled, y_train)
decision_tree.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
```

#### 4. Making Predictions and Evaluating the Models

```python
from sklearn.metrics import classification_report, confusion_matrix

models = {
    'Logistic Regression': log_reg,
    'Decision Tree': decision_tree,
    'K-Nearest Neighbors': knn,
    'Support Vector Machine': svm
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)
```

#### 5. Comparing Model Performance

```python
import numpy as np

# Initialize a DataFrame to store accuracy
accuracy_df = pd.DataFrame(columns=['Model', 'Accuracy'])

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    accuracy_df = accuracy_df.append({'Model': name, 'Accuracy': accuracy}, ignore_index=True)

print(accuracy_df)
```

#### 6. Visualizing Model Accuracies

```python
sns.barplot(x='Accuracy', y='Model', data=accuracy_df, palette='viridis')
plt.title('Model Accuracies on Iris Test Set')
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.xlim(0, 1)
plt.show()
```

---

## 8. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 2** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, you explored various **Classification Algorithms**, implemented them using Scikit-Learn, and evaluated their performance to understand their strengths and weaknesses. By working through the example project, you gained hands-on experience in building and comparing multiple classification models.

### 🔮 What’s Next?

- **Day 3: Supervised Learning – Regression Algorithms**: Dive into regression techniques like Linear Regression, Ridge, Lasso, and Elastic Net.
- **Day 4: Model Evaluation and Selection**: Learn about cross-validation, hyperparameter tuning, and model selection strategies.
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

# 📜 Summary of Day 2 📜

- **🧠 Introduction to Supervised Learning: Classification**: Gained a foundational understanding of classification tasks and their types.
- **📊 Classification Algorithms**: Explored Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM).
- **🛠️ Implementing Classification Algorithms with Scikit-Learn**: Learned how to build, train, and evaluate different classification models using Scikit-Learn.
- **📈 Model Evaluation for Classification**: Mastered evaluation metrics including accuracy, precision, recall, F1-score, confusion matrix, and ROC curves.
- **🛠️📈 Example Project: Advanced Iris Classification**: Developed a comprehensive classification pipeline using multiple algorithms to classify Iris species and compared their performance.

