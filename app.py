import streamlit as st
from components.navbar import navigation_component

def main():
    # Call the navigation component
    navigation_component()

    # Page content logic
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["home"])[0]

    if page == "home":
        st.title("Welcome to Scikit-Learn Navigator")
    elif page == "classification":
        st.title("Classification Page")
    elif page == "regression":
        st.title("Regression Page")
    elif page == "clustering":
        st.title("Clustering Page")
    elif page == "pipelines":
        st.title("Pipelines Page")
    elif page == "comparison":
        st.title("Comparison Page")
    elif page == "advanced-analysis":
        st.title("Advanced Analysis Page")
    elif page == "explainability":
        st.title("Explainability Page")
    elif page == "explore":
        st.title("Explore Page")
    elif page == "train":
        st.title("Train Page")
    elif page == "evaluate":
        st.title("Evaluate Page")
    elif page == "deploy":
        st.title("Deploy Page")
    elif page == "visualize":
        st.title("Visualize Page")
    elif page == "tutorial":
        st.title("Tutorial Page")
    else:
        st.title("404 Page Not Found")

if __name__ == "__main__":
    main()
