import streamlit as st

# Custom CSS for the horizontal navigation bar
def load_css():
    st.markdown(
        """
        <style>
        .nav-container {
            background-color: #f8f9fa;
            padding: 10px;
            text-align: center;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .nav-item {
            display: inline-block;
            margin: 0 15px;
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .nav-item:hover {
            color: #0056b3;
            text-decoration: underline;
        }
        .active {
            color: #0056b3;
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Function to render navigation
def render_navigation():
    st.markdown(
        """
        <div class="nav-container">
            <a href="/" class="nav-item">Home</a>
            <a href="?page=tutorial" class="nav-item">Tutorial</a>
            <a href="?page=explore" class="nav-item">Explore</a>
            <a href="?page=train" class="nav-item">Train</a>
            <a href="?page=evaluate" class="nav-item">Evaluate</a>
            <a href="?page=visualize" class="nav-item">Visualize</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main logic for Streamlit
def main():
    load_css()
    render_navigation()

    # Determine which page to render
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["home"])[0]

    if page == "home":
        st.title("Welcome to Awesome Scikit-Learn")
        st.write("Explore, learn, and build with Scikit-Learn.")
    elif page == "tutorial":
        st.title("Tutorial")
        st.write("Step-by-step tutorials to learn Scikit-Learn.")
    elif page == "explore":
        st.title("Explore")
        st.write("Upload and analyze datasets.")
    elif page == "train":
        st.title("Train")
        st.write("Train machine learning models interactively.")
    elif page == "evaluate":
        st.title("Evaluate")
        st.write("Evaluate model performance with metrics.")
    elif page == "visualize":
        st.title("Visualize")
        st.write("Visualize your data and model results.")
    else:
        st.title("404")
        st.write("Page not found.")

if __name__ == "__main__":
    main()
