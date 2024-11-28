import streamlit as st

def sidebar_navigation():
    st.sidebar.title("Navigation")  # Sidebar title
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

    # Add a radio button to navigate between pages
    selected_page = st.sidebar.radio("Go to", menu)

    return selected_page
