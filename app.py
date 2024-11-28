import streamlit as st
from components.navbar import navigation_component

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "home"

def main():
    # Render the navigation component
    navigation_component()

    # Map buttons to pages
    button_to_page = {
        "home": "Home",
        "classification": "Classification",
        "regression": "Regression",
        "clustering": "Clustering",
        "pipelines": "Pipelines",
        "comparison": "Comparison",
        "advanced-analysis": "Advanced Analysis",
        "explainability": "Explainability",
        "explore": "Explore",
        "train": "Train",
        "evaluate": "Evaluate",
        "deploy": "Deploy",
        "visualize": "Visualize",
        "tutorial": "Tutorial",
    }

    # Detect which button was clicked
    for key in button_to_page.keys():
        if st.button(button_to_page[key]):
            st.session_state["page"] = key

    # Render content based on the current page
    page = st.session_state["page"]
    if page == "home":
        st.title("Welcome to Scikit-Learn Navigator")
        st.write("This is the Home Page.")
    elif page == "classification":
        st.title("Classification")
        st.write("Learn and apply classification techniques.")
    elif page == "regression":
        st.title("Regression")
        st.write("Learn and apply regression techniques.")
    elif page == "clustering":
        st.title("Clustering")
        st.write("Learn and apply clustering techniques.")
    elif page == "pipelines":
        st.title("Pipelines")
        st.write("Learn to build and use pipelines.")
    elif page == "comparison":
        st.title("Comparison")
        st.write("Compare different models and approaches.")
    elif page == "advanced-analysis":
        st.title("Advanced Analysis")
        st.write("Perform advanced data and model analysis.")
    elif page == "explainability":
        st.title("Explainability")
        st.write("Understand and explain your models.")
    elif page == "explore":
        st.title("Explore")
        st.write("Explore your data interactively.")
    elif page == "train":
        st.title("Train")
        st.write("Train machine learning models.")
    elif page == "evaluate":
        st.title("Evaluate")
        st.write("Evaluate your models' performance.")
    elif page == "deploy":
        st.title("Deploy")
        st.write("Deploy your models to production.")
    elif page == "visualize":
        st.title("Visualize")
        st.write("Visualize data and model outputs.")
    elif page == "tutorial":
        st.title("Tutorial")
        st.write("Learn step-by-step with tutorials.")
    else:
        st.title("404 - Page Not Found")
        st.write("The page you are looking for does not exist.")

if __name__ == "__main__":
    main()
