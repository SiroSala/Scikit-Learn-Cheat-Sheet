import streamlit as st

def sidebar_navigation():
    # Add a title to the custom sidebar
    st.sidebar.title("Navigation")

    # Define menu options
    menu = [
        "Home",
        "Classification",
        "Regression",
        "Clustering",
        "Pipelines",
        "Comparison",
        "Advanced Analysis",
        "Explainability",
        "Explore",
        "Train",
        "Evaluate",
        "Deploy",
        "Visualize",
        "Tutorial",
    ]

    # Use radio buttons for navigation
    selected_page = st.sidebar.radio("Go to", menu)

    return selected_page
