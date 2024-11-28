import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Function to load Lottie animations from a URL
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to load animation from {url}")
        return None
    return response.json()

# Load animations
welcome_animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_puciaact.json")  # Welcome Animation
documentation_animation = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_hz0zyjyz.json")  # Documentation Animation
tutorial_animation = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_vf4pwnr6.json")  # Tutorial Animation
community_animation = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_hihsvl5o.json")  # Community Animation

# Define layout for Home Page
def layout():
    # Title and Welcome Animation
    st.title("Welcome to Awesome Scikit-Learn")
    if welcome_animation:
        st_lottie(welcome_animation, height=300, key="welcome")
    else:
        st.warning("Welcome animation could not be loaded.")

    # Introduction
    st.markdown(
        """
        ### 🚀 Empower Your Machine Learning Projects
        Welcome to the **ultimate platform** for exploring, training, and visualizing machine learning models using **Scikit-Learn**. 
        Whether you're a beginner or an advanced practitioner, this tool offers everything you need to:
        - Explore datasets with interactive visualizations.
        - Train and evaluate machine learning models.
        - Dive deep into advanced topics like pipelines, feature engineering, and explainability.
        """
    )

    # About Scikit-Learn
    st.markdown("### 📖 About Scikit-Learn")
    st.markdown(
        """
        Scikit-Learn is a powerful Python library for machine learning, built on NumPy, SciPy, and matplotlib. It is:
        - **Efficient**: Implements state-of-the-art machine learning algorithms.
        - **User-Friendly**: Offers a simple and consistent API for all models.
        - **Versatile**: Supports classification, regression, clustering, and dimensionality reduction.
        - **Community-Driven**: Continuously updated with contributions from researchers and developers worldwide.
        """
    )

    # Documentation Section
    st.markdown("### 📚 Comprehensive Scikit-Learn Documentation")
    if documentation_animation:
        st_lottie(documentation_animation, height=250, key="documentation")
    else:
        st.warning("Documentation animation could not be loaded.")
    st.markdown(
        """
        #### Explore Official Resources:
        - [**Scikit-Learn Documentation**](https://scikit-learn.org/stable/documentation.html): Full API reference and user guide.
        - [**Supervised Learning**](https://scikit-learn.org/stable/supervised_learning.html): Classification and regression techniques.
        - [**Unsupervised Learning**](https://scikit-learn.org/stable/unsupervised_learning.html): Clustering, PCA, and dimensionality reduction.
        - [**Model Evaluation Metrics**](https://scikit-learn.org/stable/modules/model_evaluation.html): Accuracy, precision, recall, and more.
        - [**Preprocessing Tools**](https://scikit-learn.org/stable/modules/preprocessing.html): Data scaling, encoding, and transformation.
        - [**Pipeline API**](https://scikit-learn.org/stable/modules/compose.html): Automate workflows.
        """
    )

    # Tutorials Section
    st.markdown("### 🧑‍🏫 Interactive Tutorials")
    if tutorial_animation:
        st_lottie(tutorial_animation, height=250, key="tutorial")
    else:
        st.warning("Tutorial animation could not be loaded.")
    st.markdown(
        """
        #### Step-by-Step Guides:
        - [**Introduction to Scikit-Learn**](https://scikit-learn.org/stable/tutorial/basic/tutorial.html): Beginner's guide to Scikit-Learn.
        - [**Data Exploration**](/Explore): Learn how to clean and analyze datasets.
        - [**Model Training**](/Train): Train classifiers, regressors, and clustering models.
        - [**Model Evaluation**](/Evaluate): Assess your model's performance with advanced metrics.
        - [**Feature Engineering**](https://scikit-learn.org/stable/modules/feature_extraction.html): Enhance your data for better results.
        - [**Advanced Tutorials**](https://scikit-learn.org/stable/tutorial/index.html): For experienced users.
        """
    )

    # Example Code
    st.markdown("### 🔍 Getting Started with Scikit-Learn")
    st.code(
        """
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
        """,
        language="python",
    )

    # Community and Resources Section
    st.markdown("### 🌐 Join the Community")
    if community_animation:
        st_lottie(community_animation, height=250, key="community")
    else:
        st.warning("Community animation could not be loaded.")
    st.markdown(
        """
        #### Connect with the Scikit-Learn Ecosystem:
        - [**GitHub Repository**](https://github.com/scikit-learn/scikit-learn): Contribute or explore the source code.
        - [**Mailing List**](https://mail.python.org/mailman/listinfo/scikit-learn): Join discussions with the Scikit-Learn community.
        - [**User Examples**](https://scikit-learn.org/stable/auto_examples/index.html): Explore curated examples.
        - [**Contributing Guide**](https://scikit-learn.org/stable/developers/contributing.html): Learn how to contribute to Scikit-Learn development.
        """
    )

    # Call to Action
    st.markdown(
        """
        ### 🚀 Ready to Get Started?
        Use the navigation bar above to explore features, tutorials, and datasets. Let's build something awesome together!
        """
    )
