import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animations
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load animations
welcome_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_uhfvjksk.json")
documentation_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_kswsvqse.json")
tutorial_animation = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_kdx6cani.json")

# Define layout for Home Page
def layout():
    # Set the page layout
    st.markdown(
        """
        <style>
        .welcome-header {
            font-size: 36px;
            font-weight: bold;
            color: #2C3E50;
            text-align: center;
            margin-top: 20px;
        }
        .section-header {
            font-size: 28px;
            font-weight: bold;
            color: #3498DB;
            margin-bottom: 10px;
        }
        .content-box {
            background-color: #F8F9F9;
            border: 1px solid #E5E8E8;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .center {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Welcome Header
    st.markdown("<div class='welcome-header'>Welcome to Awesome Scikit-Learn</div>", unsafe_allow_html=True)

    # Animation
    st_lottie(welcome_animation, height=250, key="welcome")

    # Welcome Post
    st.markdown(
        """
        <div class='content-box'>
            <h3>🚀 Empower Your Machine Learning Projects</h3>
            <p>
                Welcome to the ultimate platform for exploring, training, and visualizing machine learning models using 
                <strong>Scikit-Learn</strong>. Whether you're a beginner or a seasoned data scientist, this application provides 
                tools to build, evaluate, and deploy models with ease.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scikit-Learn Overview Section
    st.markdown("<div class='section-header'>About Scikit-Learn</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='content-box'>
            <p>
                Scikit-Learn is a free machine learning library for Python. It features:
            </p>
            <ul>
                <li>📊 A wide range of supervised and unsupervised learning algorithms</li>
                <li>⚙️ Tools for model evaluation, pipeline creation, and hyperparameter optimization</li>
                <li>📈 Comprehensive support for data preprocessing, including scaling, encoding, and transformation</li>
            </ul>
            <h4>Key Features</h4>
            <ul>
                <li><strong>Supervised Learning:</strong> Algorithms for classification and regression (e.g., Logistic Regression, SVM, Random Forests)</li>
                <li><strong>Unsupervised Learning:</strong> Clustering and dimensionality reduction (e.g., K-Means, PCA)</li>
                <li><strong>Model Pipelines:</strong> Easily combine preprocessing and modeling steps</li>
                <li><strong>Evaluation Metrics:</strong> Metrics for regression, classification, and clustering performance</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scikit-Learn Documentation Section
    st.markdown("<div class='section-header'>Scikit-Learn Documentation</div>", unsafe_allow_html=True)
    st_lottie(documentation_animation, height=150, key="documentation")
    st.markdown(
        """
        <div class='content-box'>
            <h4>🔗 <a href="https://scikit-learn.org/stable/documentation.html" target="_blank">Official Documentation</a></h4>
            <p>
                Explore the official Scikit-Learn documentation for:
            </p>
            <ul>
                <li><a href="https://scikit-learn.org/stable/supervised_learning.html" target="_blank">Supervised Learning</a> – Understand classification and regression algorithms</li>
                <li><a href="https://scikit-learn.org/stable/unsupervised_learning.html" target="_blank">Unsupervised Learning</a> – Dive into clustering and dimensionality reduction techniques</li>
                <li><a href="https://scikit-learn.org/stable/model_selection.html" target="_blank">Model Selection</a> – Learn about cross-validation and hyperparameter tuning</li>
                <li><a href="https://scikit-learn.org/stable/modules/classes.html" target="_blank">API Reference</a> – Detailed documentation for all classes and functions</li>
                <li><a href="https://scikit-learn.org/stable/user_guide.html" target="_blank">User Guide</a> – Tutorials and guides to get started</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tutorials Section
    st.markdown("<div class='section-header'>Interactive Tutorials</div>", unsafe_allow_html=True)
    st_lottie(tutorial_animation, height=150, key="tutorial")
    st.markdown(
        """
        <div class='content-box'>
            <h4>🧑‍💻 Tutorials</h4>
            <ul>
                <li><strong>Introduction:</strong> Learn the basics of Scikit-Learn</li>
                <li><strong>Data Preprocessing:</strong> Explore techniques for cleaning and transforming data</li>
                <li><strong>Model Training:</strong> Train classifiers, regressors, and clustering models</li>
                <li><strong>Model Evaluation:</strong> Assess your model's performance</li>
                <li><strong>Hyperparameter Optimization:</strong> Use GridSearchCV and RandomizedSearchCV for better results</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Call to Action
    st.markdown(
        """
        <div class='center'>
            <h3>✨ Ready to Dive In?</h3>
            <p>Use the navigation bar above to explore features and tutorials!</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
