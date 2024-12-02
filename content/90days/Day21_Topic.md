<div style="text-align: center;">
  <h1 style="color:#673AB7;">🚀 Becoming a Scikit-Learn Boss in 90 Days: Day 21 – Deep Learning Fundamentals and Integration with Scikit-Learn Pipelines 🤖🔗</h1>
  <p style="font-size:18px;">Enhance Your Machine Learning Pipelines by Seamlessly Integrating Deep Learning Models!</p>
  
  <!-- Animated Header Image -->
  <img src="https://media.giphy.com/media/26BRzozg4TCBXv6QU/giphy.gif" alt="Deep Learning Integration" width="600">
</div>

---

## 📑 Table of Contents

1. [🌟 Welcome to Day 21](#welcome-to-day-21)
2. [🔍 Review of Day 20 📜](#review-of-day-20-📜)
3. [🧠 Introduction to Deep Learning Fundamentals 🧠](#introduction-to-deep-learning-fundamentals-🧠)
    - [📚 What is Deep Learning?](#what-is-deep-learning-📚)
    - [🔍 Neural Networks Architecture](#neural-networks-architecture-🔍)
    - [🔄 Activation Functions](#activation-functions-🔄)
    - [🔄 Loss Functions](#loss-functions-🔄)
    - [🔄 Optimizers](#optimizers-🔄)
    - [🔄 Regularization Techniques](#regularization-techniques-🔄)
4. [🛠️ Integration with Scikit-Learn Pipelines 🛠️](#integration-with-scikit-learn-pipelines-🛠️)
    - [📊 Why Integrate Deep Learning with Scikit-Learn?](#why-integrate-deep-learning-with-scikit-learn-📊)
    - [🔗 Methods of Integration](#methods-of-integration-🔗)
        - [🧰 Using KerasClassifier/KerasRegressor](#using-kerasclassifierkerasregressor-🧰)
        - [🧰 Custom Transformers](#custom-transformers-🧰)
    - [📈 Building a Scikit-Learn Pipeline with CNNs](#building-a-scikit-learn-pipeline-with-cnns-📈)
5. [🛠️ Implementing Deep Learning within Scikit-Learn Pipelines 🛠️](#implementing-deep-learning-within-scikit-learn-pipelines-🛠️)
    - [🔡 Setting Up the Environment](#setting-up-the-environment-🔡)
    - [🤖 Building a Deep Learning Model with Keras](#building-a-deep-learning-model-with-keras-🤖)
    - [🧰 Wrapping the Model with KerasClassifier](#wrapping-the-model-with-kerasclassifier-🧰)
    - [📈 Creating the Pipeline](#creating-the-pipeline-📈)
    - [📊 Training and Evaluating the Pipeline](#training-and-evaluating-the-pipeline-📊)
6. [📈 Example Project: Sentiment Analysis with CNNs and Scikit-Learn Pipelines 📈](#example-project-sentiment-analysis-with-cnns-and-scikit-learn-pipelines-📈)
    - [📋 Project Overview](#project-overview-📋)
    - [📝 Step-by-Step Guide](#step-by-step-guide-📝)
        - [1. Load and Explore the Dataset](#1-load-and-explore-the-dataset)
        - [2. Data Preprocessing and Tokenization](#2-data-preprocessing-and-tokenization)
        - [3. Building the CNN Model](#3-building-the-cnn-model)
        - [4. Integrating the CNN with Scikit-Learn Pipeline](#4-integrating-the-cnn-with-scikit-learn-pipeline)
        - [5. Training the Pipeline](#5-training-the-pipeline)
        - [6. Evaluating the Model](#6-evaluating-the-model)
        - [7. Hyperparameter Tuning](#7-hyperparameter-tuning)
    - [📊 Results and Insights](#results-and-insights-📊)
7. [🚀🎓 Conclusion and Next Steps 🚀🎓](#conclusion-and-next-steps-🚀🎓)
8. [📜 Summary of Day 21 📜](#summary-of-day-21-📜)

---

## 1. 🌟 Welcome to Day 21

Welcome to **Day 21** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, we'll delve into the **Fundamentals of Deep Learning** and explore how to seamlessly **Integrate Deep Learning Models with Scikit-Learn Pipelines**. By mastering these concepts, you'll be able to enhance your machine learning workflows, leveraging the power of deep learning within the versatile framework of Scikit-Learn. This integration allows for the creation of robust, scalable, and maintainable machine learning solutions that combine traditional and deep learning techniques.

---

## 2. 🔍 Review of Day 20 📜

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Comprehensive Computer Vision Projects**: Undertook three significant projects to apply computer vision techniques:
  - **Real-Time Object Detection and Tracking**: Integrated YOLO for detection and OpenCV's tracking algorithms to maintain object identities in video streams.
  - **Medical Image Analysis for Tumor Detection**: Built a CNN model to classify MRI scans, achieving high accuracy in detecting tumors.
  - **Image Captioning with CNNs and RNNs**: Combined CNNs for feature extraction and RNNs for sequence generation to create descriptive captions for images.

With these projects, you gained hands-on experience in developing end-to-end computer vision solutions, integrating deep learning models with traditional machine learning workflows.

---

## 3. 🧠 Introduction to Deep Learning Fundamentals 🧠

### 📚 What is Deep Learning?

**Deep Learning** is a subset of machine learning that employs neural networks with multiple layers (hence "deep") to model complex patterns in data. Inspired by the human brain's structure, deep learning models excel in tasks such as image and speech recognition, natural language processing, and autonomous driving.

### 🔍 Neural Networks Architecture

Understanding the architecture of neural networks is crucial for designing effective deep learning models.

- **Neurons and Layers**:
  - **Input Layer**: Receives the raw data.
  - **Hidden Layers**: Perform computations and extract features. Multiple hidden layers enable the network to learn hierarchical representations.
  - **Output Layer**: Produces the final prediction or classification.

- **Types of Layers**:
  - **Dense (Fully Connected) Layers**: Each neuron is connected to every neuron in the previous layer.
  - **Convolutional Layers**: Apply convolutional filters to detect local patterns in data, particularly effective for image processing.
  - **Pooling Layers**: Reduce the spatial dimensions of data, decreasing computational load and controlling overfitting.
  - **Recurrent Layers**: Handle sequential data by maintaining a state that captures information about previous inputs.

### 🔄 Activation Functions

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.

- **ReLU (Rectified Linear Unit)**: Outputs the input directly if positive; otherwise, it outputs zero. Commonly used in hidden layers.
  
  ```python
  from keras.layers import Activation, Dense

  model.add(Dense(64))
  model.add(Activation('relu'))
  ```

- **Sigmoid**: Outputs values between 0 and 1, often used in binary classification.
  
  ```python
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  ```

- **Softmax**: Outputs a probability distribution over multiple classes, used in multi-class classification.
  
  ```python
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  ```

### 🔄 Loss Functions

Loss functions measure the discrepancy between the model's predictions and the actual targets, guiding the optimization process.

- **Mean Squared Error (MSE)**: Used for regression tasks.
  
  ```python
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  ```

- **Binary Cross-Entropy**: Used for binary classification.
  
  ```python
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

- **Categorical Cross-Entropy**: Used for multi-class classification.
  
  ```python
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  ```

### 🔄 Optimizers

Optimizers adjust the model's weights to minimize the loss function.

- **SGD (Stochastic Gradient Descent)**: Updates weights based on the gradient of the loss function.
  
  ```python
  from keras.optimizers import SGD

  optimizer = SGD(lr=0.01, momentum=0.9)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  ```

- **Adam (Adaptive Moment Estimation)**: Combines the advantages of two other extensions of stochastic gradient descent.
  
  ```python
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

### 🔄 Regularization Techniques

Regularization helps prevent overfitting by adding constraints to the model.

- **Dropout**: Randomly deactivates a fraction of neurons during training.
  
  ```python
  from keras.layers import Dropout

  model.add(Dropout(0.5))
  ```

- **L2 Regularization**: Adds a penalty proportional to the square of the magnitude of weights.
  
  ```python
  from keras.regularizers import l2

  model.add(Dense(64, kernel_regularizer=l2(0.01)))
  ```

---

## 4. 🛠️ Integration with Scikit-Learn Pipelines 🛠️

### 📊 Why Integrate Deep Learning with Scikit-Learn?

Integrating deep learning models with Scikit-Learn pipelines offers numerous benefits:

- **Streamlined Workflows**: Combine preprocessing, feature extraction, model training, and evaluation into a single pipeline.
- **Model Selection and Hyperparameter Tuning**: Utilize Scikit-Learn's tools like `GridSearchCV` and `RandomizedSearchCV` for efficient hyperparameter optimization.
- **Compatibility**: Merge deep learning models with traditional machine learning components, enabling hybrid approaches.
- **Reproducibility**: Create standardized and reproducible workflows for consistent results across different experiments.

### 🔗 Methods of Integration

#### 🧰 Using KerasClassifier/KerasRegressor

The `KerasClassifier` and `KerasRegressor` wrappers from `keras.wrappers.scikit_learn` allow you to integrate Keras (TensorFlow) models within Scikit-Learn pipelines.

```python
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap the model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
```

#### 🧰 Custom Transformers

Create custom transformers by subclassing `BaseEstimator` and `TransformerMixin` to include deep learning components within Scikit-Learn pipelines.

```python
from sklearn.base import BaseEstimator, TransformerMixin
from keras.models import load_model
import numpy as np

class DeepFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model_path):
        self.model = load_model(model_path)
        # Remove the output layer to use as feature extractor
        self.model = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = self.model.predict(X)
        return features
```

### 📈 Building a Scikit-Learn Pipeline with CNNs

Integrate a CNN-based feature extractor with a traditional classifier in a Scikit-Learn pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

pipeline = Pipeline([
    ('feature_extractor', DeepFeatureExtractor(model_path='cnn_model.h5')),
    ('classifier', SVC(kernel='linear'))
])

# Example usage:
# pipeline.fit(X_train, y_train)
# predictions = pipeline.predict(X_test)
```

*Note: Ensure that the deep learning model is compatible with Scikit-Learn's pipeline requirements, such as input shapes and data types.*

---

## 5. 🛠️ Implementing Deep Learning within Scikit-Learn Pipelines 🛠️

### 🔡 Setting Up the Environment 🔡

Ensure you have the necessary libraries installed.

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install required libraries
pip install scikit-learn tensorflow keras matplotlib numpy
```

### 🤖 Building a Deep Learning Model with Keras 🤖

Define and build a CNN model using Keras.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```

### 🧰 Wrapping the Model with KerasClassifier 🧰

Use `KerasClassifier` to integrate the CNN with Scikit-Learn.

```python
from keras.wrappers.scikit_learn import KerasClassifier

# Wrap the CNN Model
cnn_classifier = KerasClassifier(build_fn=create_cnn_model, epochs=20, batch_size=32, verbose=1)
```

### 📈 Creating the Pipeline 📈

Combine the CNN classifier within a Scikit-Learn pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('cnn', cnn_classifier),
    ('svc', SVC(kernel='linear'))
])

# Example: Define Parameter Grid for Grid Search
param_grid = {
    'cnn__epochs': [10, 20],
    'cnn__batch_size': [32, 64],
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf']
}
```

### 📊 Training and Evaluating the Pipeline 📊

Train the pipeline using the training data and evaluate its performance.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset (Example: CIFAR-10)
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize Pixel Values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Initialize Grid Search
grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)

# Train the Pipeline
grid.fit(X_train, y_train)

# Best Parameters
print(f"Best Parameters: {grid.best_params_}")

# Evaluate on Test Data
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

*Note: Hyperparameter tuning with deep learning models can be computationally intensive. Consider using a subset of data or leveraging cloud-based resources for efficient training.*

---

## 6. 📈 Example Project: Sentiment Analysis with CNNs and Scikit-Learn Pipelines 📈

### 📋 Project Overview

**Objective**: Develop a sentiment analysis system that classifies text data (e.g., movie reviews) as positive or negative by integrating Convolutional Neural Networks (CNNs) within Scikit-Learn pipelines. This project combines natural language processing (NLP) techniques with deep learning to analyze and interpret textual data effectively.

**Tools**: Python, Scikit-Learn, Keras (TensorFlow), NLTK, Matplotlib, NumPy

### 📝 Step-by-Step Guide

#### 1. Load and Explore the Dataset

We'll use the **IMDB Movie Reviews** dataset, which contains labeled movie reviews.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load IMDB Dataset
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 5000  # Vocabulary size
maxlen = 400         # Maximum review length

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# Explore Dataset
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")

# Decode Review Back to Text (Example)
word_index = imdb.get_word_index()
reverse_word_index = {v:k for k,v in word_index.items()}
def decode_review(text):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in text])

print("Sample Review:", decode_review(X_train[0]))
print("Sentiment:", "Positive" if y_train[0] == 1 else "Negative")
```

#### 2. Data Preprocessing and Tokenization

Prepare the text data for the CNN model.

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Pad Sequences to Ensure Uniform Length
X_train_padded = pad_sequences(X_train, maxlen=maxlen)
X_test_padded = pad_sequences(X_test, maxlen=maxlen)

print(f"Padded Training Shape: {X_train_padded.shape}")
print(f"Padded Test Shape: {X_test_padded.shape}")
```

#### 3. Building the CNN Model

Define a CNN model suitable for text classification.

```python
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout

def create_text_cnn():
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize Model
text_cnn = create_text_cnn()
text_cnn.summary()
```

#### 4. Integrating the CNN with Scikit-Learn Pipeline

Wrap the CNN model using `KerasClassifier` and create a pipeline.

```python
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Wrap the CNN Model
cnn_classifier = KerasClassifier(build_fn=create_text_cnn, epochs=5, batch_size=32, verbose=1)

# Create a Pipeline
pipeline = Pipeline([
    ('cnn', cnn_classifier)
])

# Define Parameter Grid
param_grid = {
    'cnn__epochs': [5, 10],
    'cnn__batch_size': [32, 64]
}

# Initialize Grid Search
grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
```

#### 5. Training the Pipeline

Train the pipeline using the training data.

```python
# Train the Pipeline
grid.fit(X_train_padded, y_train)

# Best Parameters
print(f"Best Parameters: {grid.best_params_}")
```

#### 6. Evaluating the Model

Assess the model's performance on the test dataset.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on Test Data
y_pred = grid.predict(X_test_padded)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Sentiment Analysis')
plt.show()
```

#### 7. Hyperparameter Tuning

Optimize model performance by tuning hyperparameters using Grid Search.

```python
# Perform Grid Search
grid.fit(X_train_padded, y_train)

# Best Parameters
print(f"Best Parameters: {grid.best_params_}")

# Evaluate Best Model on Test Data
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_padded)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Test Accuracy of Best Model: {accuracy_best:.2f}")

# Classification Report for Best Model
print("Classification Report for Best Model:")
print(classification_report(y_test, y_pred_best))
```

#### 📊 Results and Insights

The sentiment analysis system effectively classifies movie reviews with high accuracy, demonstrating the efficacy of integrating CNNs within Scikit-Learn pipelines. Hyperparameter tuning further enhances model performance, showcasing the benefits of automated model optimization within Scikit-Learn's framework.

---

## 7. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 21** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, you explored the **Fundamentals of Deep Learning** and learned how to **Integrate Deep Learning Models with Scikit-Learn Pipelines**. By mastering these concepts, you enhanced your ability to build sophisticated machine learning workflows that leverage the strengths of both traditional and deep learning techniques. Through hands-on projects like sentiment analysis, you demonstrated the practical application of integrating CNNs within Scikit-Learn pipelines, paving the way for more advanced and scalable machine learning solutions.

### 🔮 What’s Next?

- **Days 22-25: Advanced Deep Learning Techniques**
  - **Day 22**: Transfer Learning and Fine-Tuning Models
  - **Day 23**: Generative Adversarial Networks (GANs)
  - **Day 24**: Reinforcement Learning Basics
  - **Day 25**: Deploying Machine Learning Models to Production
- **Days 26-90: Specialized Topics and Comprehensive Projects**
  - Explore areas like reinforcement learning, advanced ensemble methods, model optimization, and deploying models to cloud platforms.
  - Engage in larger projects that integrate multiple machine learning techniques to solve complex real-world problems.

### 📝 Tips for Success

- **Practice Regularly**: Continuously apply the concepts through exercises, projects, and real-world applications to reinforce your learning.
- **Engage with the Community**: Participate in forums, attend webinars, and collaborate with peers to exchange knowledge and tackle challenges together.
- **Stay Curious**: Keep exploring new features, updates, and best practices in Scikit-Learn, TensorFlow, Keras, and the broader machine learning ecosystem.
- **Document Your Work**: Maintain a detailed journal or portfolio of your projects and learning milestones to track your progress and showcase your skills to potential employers or collaborators.

Keep up the excellent work, and stay motivated as you continue your journey to mastering Scikit-Learn and becoming a proficient machine learning practitioner! 🚀📚


---

# 📜 Summary of Day 21 📜

- **🧠 Introduction to Deep Learning Fundamentals**: Gained a comprehensive understanding of deep learning, neural network architectures, activation functions, loss functions, optimizers, and regularization techniques.
- **🔗 Integration with Scikit-Learn Pipelines**: Learned methods to seamlessly incorporate deep learning models within Scikit-Learn pipelines using `KerasClassifier` and custom transformers, enabling streamlined workflows and efficient model optimization.
- **📊 Building a Scikit-Learn Pipeline with CNNs**: Constructed a cohesive pipeline that combines CNN-based feature extraction with traditional machine learning classifiers, enhancing model robustness and scalability.
- **🛠️ Implementing Deep Learning within Scikit-Learn Pipelines**: Developed and trained CNN models, wrapped them for Scikit-Learn compatibility, and evaluated their performance within integrated pipelines.
- **📈 Example Project: Sentiment Analysis with CNNs and Scikit-Learn Pipelines**: Built a sentiment analysis system using the IMDB dataset, demonstrating the practical application of integrating CNNs within Scikit-Learn pipelines for text classification tasks.
- **🛠️📈 Practical Skills Acquired**: Enhanced ability to integrate deep learning models with Scikit-Learn, build robust machine learning pipelines, and optimize models for improved performance through hyperparameter tuning.
