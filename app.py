import streamlit as st
from components.navbar import sidebar_navigation

def main():
    # Render the sidebar and get the selected page
    selected_page = sidebar_navigation()

    # Display content based on the selected page
    if selected_page == "Home":
        st.title("Home")
        st.write("Welcome to the Scikit-Learn Navigator Home!")
    elif selected_page == "Classification":
        st.title("Classification")
        st.write("Learn and apply classification techniques.")
    elif selected_page == "Regression":
        st.title("Regression")
        st.write("Learn and apply regression techniques.")
    elif selected_page == "Clustering":
        st.title("Clustering")
        st.write("Learn and apply clustering techniques.")
    elif selected_page == "Pipelines":
        st.title("Pipelines")
        st.write("Learn to build and use pipelines.")
    elif selected_page == "Comparison":
        st.title("Comparison")
        st.write("Compare different models and approaches.")
    elif selected_page == "Advanced Analysis":
        st.title("Advanced Analysis")
        st.write("Perform advanced data and model analysis.")
    elif selected_page == "Explainability":
        st.title("Explainability")
        st.write("Understand and explain your models.")
    elif selected_page == "Explore":
        st.title("Explore")
        st.write("Explore your data interactively.")
    elif selected_page == "Train":
        st.title("Train")
        st.write("Train machine learning models.")
    elif selected_page == "Evaluate":
        st.title("Evaluate")
        st.write("Evaluate your models' performance.")
    elif selected_page == "Deploy":
        st.title("Deploy")
        st.write("Deploy your models to production.")
    elif selected_page == "Visualize":
        st.title("Visualize")
        st.write("Visualize data and model outputs.")
    elif selected_page == "Tutorial":
        st.title("Tutorial")
        st.write("Learn step-by-step with tutorials.")
    else:
        st.title("404 - Page Not Found")
        st.write("The page you are looking for does not exist.")

if __name__ == "__main__":
    main()
