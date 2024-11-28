# Ensure set_page_config is the very first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="Awesome Scikit-Learn",
    page_icon="🔍",
    layout="wide"
)

from streamlit_lottie import st_lottie
import requests
from components.footer import display_footer  # Import footer module

# Function to load Lottie animations from a URL
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load animations
welcome_animation = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_puciaact.json")
documentation_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_1pxqjqps.json")
tutorial_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_1a8dx7zj.json")
community_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_8wREpI.json")

# Define the layout for Home Page
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
        - [**Official Documentation**](https://scikit-learn.org/stable/documentation.html)
        - [**Supervised Learning**](https://scikit-learn.org/stable/supervised_learning.html)
        - [**Unsupervised Learning**](https://scikit-learn.org/stable/unsupervised_learning.html)
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
        - [**Introduction to Scikit-Learn**](https://scikit-learn.org/stable/tutorial/basic/tutorial.html)
        - [**Data Exploration**](/Explore)
        - [**Model Training**](/Train)
        """
    )

    # Community Section
    st.markdown("### 🌐 Join the Community")
    if community_animation:
        st_lottie(community_animation, height=250, key="community")
    else:
        st.warning("Community animation could not be loaded.")
    st.markdown(
        """
        - [**GitHub Repository**](https://github.com/scikit-learn/scikit-learn)
        - [**Mailing List**](https://mail.python.org/mailman/listinfo/scikit-learn)
        """
    )

    # Footer
    display_footer()
