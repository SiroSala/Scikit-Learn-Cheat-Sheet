<div style="text-align: center;">
  <h1>📝 Day 7: Natural Language Processing (NLP) – Text Processing, Sentiment Analysis, and Language Models 🧠📚</h1>
  <p>Unlock the Power of Text Data to Derive Meaningful Insights!</p>
</div>

---

## 📑 Table of Contents

1. [📅 Review of Day 6 📜](#review-of-day-6-📜)
2. [🧠 Introduction to Natural Language Processing (NLP)](#introduction-to-natural-language-processing-nlp-🧠)
    - [🔍 What is NLP?](#what-is-nlp-🔍)
    - [🛠️ NLP Applications](#nlp-applications-🛠️)
    - [📚 NLP Libraries Overview](#nlp-libraries-overview-📚)
3. [✂️ Text Preprocessing](#text-preprocessing-✂️)
    - [🧹 Cleaning Text Data](#cleaning-text-data-🧹)
    - [🔤 Tokenization](#tokenization-🔤)
    - [🔡 Removing Stop Words](#removing-stop-words-🔡)
    - [🔄 Stemming and Lemmatization](#stemming-and-lemmatization-🔄)
4. [📊 Feature Extraction for Text Data](#feature-extraction-for-text-data-📊)
    - [📈 Bag of Words (BoW)](#bag-of-words-bow-📈)
    - [🔢 Term Frequency-Inverse Document Frequency (TF-IDF)](#term-frequency-inverse-document-frequency-tf-idf-🔢)
    - [🧩 Word Embeddings](#word-embeddings-🧩)
5. [🔍 Sentiment Analysis](#sentiment-analysis-🔍)
    - [📚 Understanding Sentiment Analysis](#understanding-sentiment-analysis-📚)
    - [📈 Building a Sentiment Analysis Model](#building-a-sentiment-analysis-model-📈)
    - [📉 Evaluating Sentiment Models](#evaluating-sentiment-models-📉)
6. [🧩 Language Models and Transformers](#language-models-and-transformers-🧩)
    - [🔧 Introduction to Transformers](#introduction-to-transformers-🔧)
    - [🤖 Using Pre-trained Models with Hugging Face](#using-pre-trained-models-with-hugging-face-🤖)
    - [📈 Fine-Tuning Language Models](#fine-tuning-language-models-📈)
7. [🛠️📈 Example Project: Sentiment Analysis on Movie Reviews](#example-project-sentiment-analysis-on-movie-reviews-🛠️📈)
8. [🚀🎓 Conclusion and Next Steps](#conclusion-and-next-steps-🚀🎓)
9. [📜 Summary of Day 7 📜](#summary-of-day-7-📜)

---

## 1. 📅 Review of Day 6 📜

Before diving into today's topics, let's briefly recap what we covered yesterday:

- **Deep Learning Basics**: Explored neural networks, TensorFlow, and Keras.
- **Convolutional Neural Networks (CNNs)**: Learned about CNN architectures for image processing.
- **Recurrent Neural Networks (RNNs)**: Delved into RNNs for sequential data.
- **Transfer Learning**: Leveraged pre-trained models to enhance performance.
- **Example Project**: Built and trained a neural network to classify the MNIST dataset.

With this foundation, we're ready to explore Natural Language Processing (NLP), a crucial area for analyzing and deriving insights from text data.

---

## 2. 🧠 Introduction to Natural Language Processing (NLP) 🧠

Natural Language Processing (NLP) is a field at the intersection of computer science, artificial intelligence, and linguistics. It focuses on enabling computers to understand, interpret, and generate human language.

### 🔍 What is NLP?

**NLP** involves the interaction between computers and humans through natural language. The ultimate goal is to read, decipher, understand, and make sense of human languages in a valuable way.

### 🛠️ NLP Applications

- **Sentiment Analysis**: Determining the sentiment expressed in text.
- **Machine Translation**: Translating text from one language to another.
- **Chatbots and Virtual Assistants**: Interacting with users in natural language.
- **Text Summarization**: Creating concise summaries of longer texts.
- **Named Entity Recognition (NER)**: Identifying and classifying key information in text.

### 📚 NLP Libraries Overview

- **NLTK (Natural Language Toolkit)**: A comprehensive library for building Python programs to work with human language data.
- **SpaCy**: An industrial-strength NLP library for Python, optimized for performance.
- **Gensim**: A library for topic modeling and document similarity analysis.
- **Hugging Face Transformers**: A library providing state-of-the-art pre-trained models for NLP tasks.

---

## 3. ✂️ Text Preprocessing ✂️

Text preprocessing is a crucial step in NLP, involving cleaning and transforming raw text into a format suitable for analysis.

### 🧹 Cleaning Text Data 🧹

Cleaning involves removing noise from the text data to improve the quality of analysis.

```python
import re

def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

sample_text = "Hello World! This is a test text, with numbers 123 and symbols #@$."
cleaned = clean_text(sample_text)
print(cleaned)
```

### 🔤 Tokenization 🔤

Tokenization is the process of breaking down text into individual words or tokens.

```python
from nltk.tokenize import word_tokenize

text = "Natural Language Processing is fascinating."
tokens = word_tokenize(text)
print(tokens)
```

### 🔡 Removing Stop Words 🔡

Stop words are commonly used words that carry minimal meaningful information.

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
words = word_tokenize("This is a simple example demonstrating stop words removal.")
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)
```

### 🔄 Stemming and Lemmatization 🔄

Stemming and lemmatization reduce words to their base or root form.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

word = "running"
stemmed = stemmer.stem(word)
lemmatized = lemmatizer.lemmatize(word, pos='v')
print(f"Stemmed: {stemmed}, Lemmatized: {lemmatized}")
```

---

## 4. 📊 Feature Extraction for Text Data 📊

Transforming text into numerical representations is essential for machine learning models.

### 📈 Bag of Words (BoW) 📈

BoW represents text by the frequency of each word in the document.

```python
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love machine learning.",
    "Machine learning is amazing.",
    "I enjoy learning new things."
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 🔢 Term Frequency-Inverse Document Frequency (TF-IDF) 🔢

TF-IDF weighs the frequency of a word in a document against its rarity across all documents.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 🧩 Word Embeddings 🧩

Word embeddings capture semantic relationships between words in a continuous vector space.

```python
import gensim.downloader as api

# Load pre-trained word vectors
model = api.load("glove-wiki-gigaword-50")

# Get the vector for a word
vector = model['machine']
print(vector)
```

---

## 5. 🔍 Sentiment Analysis 🔍

Sentiment Analysis involves determining the emotional tone behind a body of text.

### 📚 Understanding Sentiment Analysis 📚

Sentiment Analysis classifies text into predefined categories such as positive, negative, or neutral.

### 📈 Building a Sentiment Analysis Model 📈

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample DataFrame
data = {
    'Review': [
        "I loved the movie! It was fantastic.",
        "The movie was terrible and boring.",
        "An average film with some good moments.",
        "Absolutely wonderful! A masterpiece.",
        "Not good, not bad. It was okay."
    ],
    'Sentiment': [1, 0, 1, 1, 0]  # 1: Positive, 0: Negative
}
df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluate
print(classification_report(y_test, y_pred))
```

### 📉 Evaluating Sentiment Models 📉

Use metrics like accuracy, precision, recall, and F1-score to assess model performance.

---

## 6. 🧩 Language Models and Transformers 🧩

Language Models predict the probability of a sequence of words and are fundamental to many NLP tasks.

### 🔧 Introduction to Transformers 🔧

**Transformers** are a type of model architecture that relies on self-attention mechanisms, enabling parallel processing of data and capturing long-range dependencies.

### 🤖 Using Pre-trained Models with Hugging Face 🤖

Hugging Face provides a vast repository of pre-trained transformer models.

```python
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Analyze sentiment
result = sentiment_pipeline("I love using Hugging Face models!")
print(result)
```

### 📈 Fine-Tuning Language Models 📈

Fine-tuning involves adapting a pre-trained model to a specific task using task-specific data.

```python
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer
import pandas as pd

# Sample DataFrame
data = {
    'text': ["I love this!", "This is bad."],
    'label': [1, 0]
}
df = pd.DataFrame(data)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize data
inputs = tokenizer(list(df['text']), padding=True, truncation=True, return_tensors="pt")
labels = df['label'].tolist()

# Define dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

dataset = SentimentDataset(inputs, labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train the model
trainer.train()
```

---

## 7. 🛠️📈 Example Project: Sentiment Analysis on Movie Reviews 🛠️📈

Let's apply today's concepts by building a sentiment analysis model to classify movie reviews as positive or negative.

### 📋 Project Overview

**Objective**: Develop a sentiment analysis model using the IMDb movie reviews dataset to classify reviews as positive or negative.

**Tools**: Python, NLTK, SpaCy, scikit-learn, Hugging Face Transformers

### 📝 Step-by-Step Guide

#### 1. Load and Explore the Dataset

```python
import pandas as pd

# Load dataset
df = pd.read_csv('IMDB_Dataset.csv')  # Ensure the dataset is downloaded and placed in your working directory
print(df.head())
```

#### 2. Text Preprocessing

```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['Cleaned_Review'] = df['review'].apply(preprocess)
print(df.head())
```

#### 3. Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Cleaned_Review'])
y = df['sentiment'].map({'positive':1, 'negative':0})
```

#### 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### 5. Build and Train the Model

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

#### 6. Using a Pre-trained Transformer Model

```python
from transformers import pipeline

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Analyze a sample review
sample_review = "I absolutely loved this movie! The performances were outstanding."
result = sentiment_pipeline(sample_review)
print(result)
```

---

## 8. 🚀🎓 Conclusion and Next Steps 🚀🎓

Congratulations on completing **Day 7**! Today, you delved into the fascinating world of Natural Language Processing (NLP), mastering text preprocessing techniques, feature extraction methods, sentiment analysis, and leveraging advanced language models with Transformers. Additionally, you built a sentiment analysis model to classify movie reviews, reinforcing your understanding through practical application.

### 🔮 What’s Next?

- **Day 8: Big Data Tools**: Introduction to Hadoop, Spark, and other big data technologies.
- **Day 9: Model Deployment and Serving**: Learn advanced deployment strategies for machine learning models.
- **Day 10: Time Series Analysis**: Explore techniques for analyzing and forecasting time-dependent data.
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

# 📜 Summary of Day 7 📜

- **🧠 Introduction to Natural Language Processing (NLP)**: Gained a foundational understanding of NLP concepts and applications.
- **✂️ Text Preprocessing**: Learned techniques for cleaning, tokenizing, removing stop words, and lemmatizing text data.
- **📊 Feature Extraction for Text Data**: Explored Bag of Words, TF-IDF, and Word Embeddings for converting text into numerical representations.
- **🔍 Sentiment Analysis**: Built and evaluated a sentiment analysis model to classify movie reviews.
- **🧩 Language Models and Transformers**: Understood transformer architectures and utilized pre-trained models with Hugging Face for NLP tasks.
- **🛠️📈 Example Project**: Developed a sentiment analysis project on movie reviews, integrating preprocessing, feature extraction, model building, and evaluation.

This structured approach ensures that you build a robust foundation in Natural Language Processing, equipping you with the skills needed to analyze and derive insights from textual data. Continue experimenting with the provided tools and don't hesitate to delve into additional resources to deepen your expertise.

**Happy Learning! 🎉**
