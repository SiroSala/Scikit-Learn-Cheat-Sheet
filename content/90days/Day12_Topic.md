<div style="text-align: center;">
  <h1>🚀 Becoming a Scikit-Learn Boss in 90 Days: Day 12 – Named Entity Recognition (NER) and Part-of-Speech (POS) Tagging 🧩🔠</h1>
  <p>Enhance Your NLP Skills by Extracting Meaningful Information and Understanding Text Structure!</p>
</div>

---

## 📑 Table of Contents

1. [🌟 Welcome to Day 12](#welcome-to-day-12)
2. [🔍 Review of Day 11 📜](#review-of-day-11-📜)
3. [🧠 Introduction to NER and POS Tagging 🧠](#introduction-to-ner-and-pos-tagging-🧠)
    - [📚 What is Part-of-Speech (POS) Tagging?](#what-is-part-of-speech-pos-tagging-📚)
    - [🔍 What is Named Entity Recognition (NER)?](#what-is-named-entity-recognition-ner-🔍)
    - [🔄 Relationship Between POS Tagging and NER](#relationship-between-pos-tagging-and-ner-🔄)
4. [🛠️ Techniques for POS Tagging and NER 🛠️](#techniques-for-pos-tagging-and-ner-🛠️)
    - [🔡 POS Tagging Techniques](#pos-tagging-techniques-🔡)
        - [📄 Rule-Based Methods](#rule-based-methods-📄)
        - [📈 Statistical Models](#statistical-models-📈)
        - [🧠 Neural Network Approaches](#neural-network-approaches-🧠)
    - [🔍 NER Techniques](#ner-techniques-🔍)
        - [📄 Rule-Based Methods](#rule-based-methods-📄-1)
        - [📈 Statistical Models](#statistical-models-📈-1)
        - [🧠 Neural Network Approaches](#neural-network-approaches-🧠-1)
5. [🛠️ Implementing POS Tagging and NER with Python 🛠️](#implementing-pos-tagging-and-ner-with-python-🛠️)
    - [🔡 POS Tagging with NLTK](#pos-tagging-with-nltk-🔡)
    - [🔍 NER with spaCy](#ner-with-spacy-🔍)
    - [📊 Integrating POS and NER with Scikit-Learn](#integrating-pos-and-ner-with-scikit-learn-📊)
6. [🛠️📈 Example Project: Enhancing Sentiment Analysis with POS and NER 🛠️📈](#example-project-enhancing-sentiment-analysis-with-pos-and-ner-🛠️📈)
    - [📋 Project Overview](#project-overview-📋)
    - [📝 Step-by-Step Guide](#step-by-step-guide-📝)
        - [1. Load and Explore the Dataset](#1-load-and-explore-the-dataset)
        - [2. Data Preprocessing](#2-data-preprocessing)
        - [3. POS Tagging](#3-pos-tagging)
        - [4. Named Entity Recognition](#4-named-entity-recognition)
        - [5. Feature Engineering with POS and NER](#5-feature-engineering-with-pos-and-ner)
        - [6. Building and Training the Sentiment Analysis Model](#6-building-and-training-the-sentiment-analysis-model)
        - [7. Evaluating Model Performance](#7-evaluating-model-performance)
    - [📊 Results and Insights](#results-and-insights-📊)
7. [🚀🎓 Conclusion and Next Steps 🚀🎓](#conclusion-and-next-steps-🚀🎓)
8. [📜 Summary of Day 12 📜](#summary-of-day-12-📜)

---

## 1. 🌟 Welcome to Day 12

Welcome to **Day 12** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, we'll delve deeper into **Natural Language Processing (NLP)** by exploring **Part-of-Speech (POS) Tagging** and **Named Entity Recognition (NER)**. These techniques are essential for understanding the structure and meaning of text, enabling more sophisticated text analysis and machine learning applications.

---

## 2. 🔍 Review of Day 11 📜

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Natural Language Processing (NLP)**: Explored the fundamentals of NLP, including text preprocessing, feature extraction techniques like Bag of Words (BoW) and TF-IDF, and built a sentiment analysis model.
- **Feature Extraction**: Learned how to convert textual data into numerical representations suitable for machine learning models.
- **Example Project**: Developed a sentiment analysis pipeline using Logistic Regression and Random Forest models, implemented feature engineering, and evaluated model performance.

With this foundation, we're now ready to enhance our NLP capabilities by understanding and implementing POS Tagging and NER.

---

## 3. 🧠 Introduction to NER and POS Tagging 🧠

### 📚 What is Part-of-Speech (POS) Tagging? 📚

**Part-of-Speech (POS) Tagging** is the process of assigning grammatical categories (such as noun, verb, adjective) to each word in a sentence. POS tagging helps in understanding the grammatical structure of sentences, which is crucial for various NLP tasks like parsing, information extraction, and sentiment analysis.

**Example:**

- Sentence: "The quick brown fox jumps over the lazy dog."
- POS Tags: The (DET) quick (ADJ) brown (ADJ) fox (NOUN) jumps (VERB) over (ADP) the (DET) lazy (ADJ) dog (NOUN).

### 🔍 What is Named Entity Recognition (NER)? 🔍

**Named Entity Recognition (NER)** is the task of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, dates, etc. NER is vital for information extraction, question answering, and building knowledge graphs.

**Example:**

- Sentence: "Apple Inc. was founded by Steve Jobs in Cupertino."
- NER Tags: Apple Inc. (ORGANIZATION), Steve Jobs (PERSON), Cupertino (LOCATION).

### 🔄 Relationship Between POS Tagging and NER 🔄

POS tagging and NER are complementary NLP techniques. While POS tagging provides grammatical context, NER identifies specific entities within that context. Combining both can enhance the performance of downstream NLP tasks by providing a richer understanding of the text.

---

## 4. 🛠️ Techniques for POS Tagging and NER 🛠️

### 🔡 POS Tagging Techniques 🔡

#### 📄 Rule-Based Methods 📄

Rule-based POS taggers use a set of linguistic rules to assign tags to words. These rules consider the word's context, suffixes, and other grammatical cues.

- **Pros**: Transparent, easy to understand and modify.
- **Cons**: Limited scalability, struggles with ambiguous words and complex sentences.

#### 📈 Statistical Models 📈

Statistical POS taggers rely on probabilistic models, such as Hidden Markov Models (HMM) and Conditional Random Fields (CRF), to predict tags based on the likelihood of sequences.

- **Pros**: More accurate than rule-based methods, can handle ambiguity better.
- **Cons**: Requires large annotated datasets for training, less interpretable than rule-based methods.

#### 🧠 Neural Network Approaches 🧠

Modern POS taggers utilize deep learning techniques, such as Recurrent Neural Networks (RNN) and Transformers, to capture complex dependencies and contextual information.

- **Pros**: High accuracy, handles long-range dependencies, adaptable to various languages.
- **Cons**: Requires substantial computational resources and data, less interpretable.

### 🔍 NER Techniques 🔍

#### 📄 Rule-Based Methods 📄

Similar to POS tagging, rule-based NER systems use predefined patterns and dictionaries to identify entities.

- **Pros**: High precision for well-defined entities, easy to implement for specific domains.
- **Cons**: Low recall, struggles with unseen entities and variations in text.

#### 📈 Statistical Models 📈

Statistical NER systems use probabilistic models like CRF and HMM to recognize entities based on context and word patterns.

- **Pros**: Better recall than rule-based methods, can generalize to new entities.
- **Cons**: Requires annotated data, sensitive to feature engineering.

#### 🧠 Neural Network Approaches 🧠

State-of-the-art NER systems leverage deep learning models, including BiLSTM-CRF and Transformer-based architectures like BERT, to achieve high performance.

- **Pros**: Superior accuracy, ability to capture contextual nuances, adaptable to multiple languages and domains.
- **Cons**: Computationally intensive, requires large datasets for training.

---

## 5. 🛠️ Implementing POS Tagging and NER with Python 🛠️

### 🔡 POS Tagging with NLTK 🔡

The Natural Language Toolkit (NLTK) provides robust tools for POS tagging using pre-trained models.

```python
import nltk
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Sample Text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text
tokens = word_tokenize(text)

# Perform POS Tagging
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
```

**Output:**
```
[('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN'), ('.', '.')]
```

### 🔍 NER with spaCy 🔍

spaCy is a powerful NLP library that provides efficient NER capabilities out of the box.

```python
import spacy

# Load spaCy's pre-trained model
nlp = spacy.load('en_core_web_sm')

# Sample Text
text = "Apple Inc. was founded by Steve Jobs in Cupertino."

# Process the text
doc = nlp(text)

# Extract Named Entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

**Output:**
```
Apple Inc. ORG
Steve Jobs PERSON
Cupertino GPE
```

### 📊 Integrating POS and NER with Scikit-Learn 📊

While Scikit-Learn doesn't provide native tools for POS tagging and NER, these techniques can be integrated into Scikit-Learn pipelines by using external libraries like NLTK or spaCy for preprocessing and feature extraction.

**Example Workflow:**

1. **Text Preprocessing**: Use NLTK or spaCy for POS tagging and NER.
2. **Feature Extraction**: Incorporate POS tags and NER entities as features using Scikit-Learn's feature extraction tools.
3. **Model Training**: Train machine learning models using the enriched feature set.

```python
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import spacy
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Custom Transformer for POS Tags
class POSTagger(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pos_tags = []
        for text in X:
            tokens = word_tokenize(text)
            tags = nltk.pos_tag(tokens)
            pos = ' '.join([tag for word, tag in tags])
            pos_tags.append(pos)
        return pos_tags

# Custom Transformer for NER
class NERTagger(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ner_tags = []
        for text in X:
            doc = nlp(text)
            entities = ' '.join([ent.label_ for ent in doc.ents])
            ner_tags.append(entities)
        return ner_tags

# Define the Pipeline
pipeline = Pipeline([
    ('pos', POSTagger()),
    ('ner', NERTagger()),
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Sample Data
X = [
    "Apple Inc. was founded by Steve Jobs.",
    "The quick brown fox jumps over the lazy dog.",
    "Microsoft Corporation released Windows 10 in 2015."
]
y = [1, 0, 1]  # 1: Organization Mentioned, 0: Not Mentioned

# Train the Model
pipeline.fit(X, y)

# Make Predictions
predictions = pipeline.predict(X)
print(predictions)
```

**Output:**
```
[1 0 1]
```

---

## 6. 🛠️📈 Example Project: Enhancing Sentiment Analysis with POS and NER 🛠️📈

### 📋 Project Overview

**Objective**: Improve a sentiment analysis model by incorporating Part-of-Speech (POS) tags and Named Entity Recognition (NER) as additional features. This enhancement aims to provide more context and nuanced understanding of the text, leading to better model performance.

**Tools**: Python, Scikit-Learn, NLTK, spaCy, pandas, NumPy, Matplotlib, Seaborn

### 📝 Step-by-Step Guide

#### 1. Load and Explore the Dataset

We'll use the IMDB Movie Reviews dataset for sentiment analysis.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
# Assume 'imdb_reviews.csv' has two columns: 'Review' and 'Sentiment' (1: Positive, 0: Negative)
df = pd.read_csv('imdb_reviews.csv')
print(df.head())

# Check class distribution
sns.countplot(x='Sentiment', data=df)
plt.title('Class Distribution')
plt.show()

# Display sample reviews
print(df.sample(5))
```

#### 2. Data Preprocessing

Clean the text data and prepare for feature extraction.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.tokenize import word_tokenize
import spacy

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize Lemmatizer and Stop Words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text Cleaning Function
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove non-letters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply Cleaning
df['Cleaned_Review'] = df['Review'].apply(clean_text)
print(df[['Review', 'Cleaned_Review']].head())
```

#### 3. POS Tagging

Extract POS tags and use them as additional features.

```python
from sklearn.base import TransformerMixin
from nltk import pos_tag

# Custom Transformer for POS Tags
class POSTagger(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        pos_tags = []
        for text in X:
            tokens = word_tokenize(text)
            tags = pos_tag(tokens)
            pos = ' '.join([tag for word, tag in tags])
            pos_tags.append(pos)
        return pos_tags

# Example Usage
pos_tagger = POSTagger()
pos_features = pos_tagger.transform(df['Cleaned_Review'])
print(pos_features[:5])
```

#### 4. Named Entity Recognition

Extract NER entities and use them as additional features.

```python
# Custom Transformer for NER
class NERTagger(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ner_tags = []
        for text in X:
            doc = nlp(text)
            entities = ' '.join([ent.label_ for ent in doc.ents])
            ner_tags.append(entities)
        return ner_tags

# Example Usage
ner_tagger = NERTagger()
ner_features = ner_tagger.transform(df['Cleaned_Review'])
print(ner_features[:5])
```

#### 5. Feature Engineering with POS and NER

Integrate POS and NER features into the feature set.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Define Feature Extraction for POS and NER
pos_transformer = Pipeline([
    ('pos', POSTagger()),
    ('tfidf', TfidfVectorizer())
])

ner_transformer = Pipeline([
    ('ner', NERTagger()),
    ('tfidf', TfidfVectorizer())
])

# Combine All Features
preprocessor = ColumnTransformer([
    ('tfidf', TfidfVectorizer(stop_words='english'), 'Cleaned_Review'),
    ('pos', pos_transformer, 'Cleaned_Review'),
    ('ner', ner_transformer, 'Cleaned_Review')
])

# Define the Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(max_iter=1000))
])

# Split the Data
from sklearn.model_selection import train_test_split

X = df[['Cleaned_Review']]
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
pipeline.fit(X_train, y_train)
```

#### 6. Building and Training the Sentiment Analysis Model

Train the model using the enriched feature set.

```python
# Already integrated into the pipeline above
# Train the model
pipeline.fit(X_train, y_train)
```

#### 7. Evaluating Model Performance

Assess the performance of the enhanced sentiment analysis model.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Make Predictions
y_pred = pipeline.predict(X_test)

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Detailed Classification Report
print(classification_report(y_test, y_pred))
```

**Sample Output:**
```
Accuracy: 0.88
Precision: 0.87
Recall: 0.90
F1 Score: 0.88

              precision    recall  f1-score   support

           0       0.89      0.85      0.87       200
           1       0.87      0.90      0.88       200

    accuracy                           0.88       400
   macro avg       0.88      0.88      0.88       400
weighted avg       0.88      0.88      0.88       400
```

---

## 7. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 12** of "Becoming a Scikit-Learn Boss in 90 Days"! Today, you advanced your **Natural Language Processing (NLP)** skills by mastering **Part-of-Speech (POS) Tagging** and **Named Entity Recognition (NER)**. By integrating these techniques into your machine learning pipelines, you enhanced your models' ability to understand and interpret textual data, leading to more accurate and insightful predictions.

### 🔮 What’s Next?

- **Days 13-15: Natural Language Processing with Advanced Techniques**
  - **Day 13**: Named Entity Recognition (NER) and Part-of-Speech (POS) Tagging (Already covered)
  - **Day 14**: Topic Modeling with Latent Dirichlet Allocation (LDA)
  - **Day 15**: Text Generation and Language Models
- **Days 16-20: Computer Vision using Scikit-Learn and Integration with Deep Learning Libraries**
- **Days 21-25: Deep Learning Fundamentals and Integration with Scikit-Learn Pipelines**
- **Days 26-90: Specialized Topics and Comprehensive Projects**
  - Explore areas like reinforcement learning, advanced ensemble methods, model optimization, and deploying models to cloud platforms.
  - Engage in larger projects that integrate multiple machine learning techniques to solve complex real-world problems.

### 📝 Tips for Success

- **Practice Regularly**: Continuously apply the concepts through exercises, projects, and real-world applications to reinforce your learning.
- **Engage with the Community**: Participate in forums, attend webinars, and collaborate with peers to exchange knowledge and tackle challenges together.
- **Stay Curious**: Keep exploring new features, updates, and best practices in Scikit-Learn and the broader machine learning ecosystem.
- **Document Your Work**: Maintain a detailed journal or portfolio of your projects and learning milestones to track your progress and showcase your skills to potential employers or collaborators.

Keep up the excellent work, and stay motivated as you continue your journey to mastering Scikit-Learn and becoming a proficient machine learning practitioner! 🚀📚

---

<div style="text-align: center;">
  <p>✨ Keep Learning, Keep Growing! ✨</p>
  <p>🚀 Your Data Science Journey Continues 🚀</p>
  <p>📚 Happy Coding! 🎉</p>
</div>

---

# 📜 Summary of Day 12 📜

- **🧠 Introduction to NER and POS Tagging**: Gained a comprehensive understanding of Part-of-Speech (POS) Tagging and Named Entity Recognition (NER) and their roles in NLP.
- **🔡 POS Tagging Techniques**: Explored rule-based, statistical, and neural network approaches for POS tagging, understanding their strengths and limitations.
- **🔍 NER Techniques**: Learned about various NER methods including rule-based, statistical models, and neural network approaches, emphasizing their applications and benefits.
- **🛠️ Implementing POS Tagging and NER with Python**: Applied POS tagging using NLTK and NER using spaCy, integrating these techniques into Scikit-Learn pipelines for enhanced feature extraction.
- **📊 Integrating POS and NER with Scikit-Learn**: Combined POS tags and NER entities with traditional text features to build a more robust sentiment analysis model.
- **🛠️📈 Example Project: Enhancing Sentiment Analysis with POS and NER**: Developed a sentiment analysis model that leverages POS tagging and NER for improved accuracy, demonstrating the practical benefits of interpretability and feature enrichment.
