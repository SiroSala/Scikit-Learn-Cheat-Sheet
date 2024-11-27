import streamlit as st

# Custom CSS for the centered navigation bar with a logo
def load_css():
    st.markdown(
        """
        <style>
        .navbar {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #f8f9fa;
            padding: 10px 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .navbar-logo {
            display: flex;
            align-items: center;
            margin-right: 20px;
        }
        .navbar-logo img {
            height: 50px;
            margin-right: 10px;
        }
        .navbar-links {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
        }
        .navbar-links a {
            margin: 0 15px;
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .navbar-links a:hover {
            color: #0056b3;
            text-decoration: underline;
        }
        .navbar-links .active {
            color: #0056b3;
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Render the navigation bar with the logo
def render_navigation(selected_page):
    st.markdown(
        f"""
        <div class="navbar">
            <div class="navbar-logo">
                <img src="https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png" alt="Logo">
                <h2 style="margin: 0; font-size: 24px; color: #2c3e50;">Awesome Scikit-Learn</h2>
            </div>
            <div class="navbar-links">
                <a href="?page=home" class="{'active' if selected_page == 'Home' else ''}">Home</a>
                <a href="?page=tutorial" class="{'active' if selected_page == 'Tutorial' else ''}">Tutorial</a>
                <a href="?page=explore" class="{'active' if selected_page == 'Explore' else ''}">Explore</a>
                <a href="?page=train" class="{'active' if selected_page == 'Train' else ''}">Train</a>
                <a href="?page=evaluate" class="{'active' if selected_page == 'Evaluate' else ''}">Evaluate</a>
                <a href="?page=visualize" class="{'active' if selected_page == 'Visualize' else ''}">Visualize</a>
                <a href="?page=classification" class="{'active' if selected_page == 'Classification' else ''}">Classification</a>
                <a href="?page=regression" class="{'active' if selected_page == 'Regression' else ''}">Regression</a>
                <a href="?page=clustering" class="{'active' if selected_page == 'Clustering' else ''}">Clustering</a>
                <a href="?page=pipelines" class="{'active' if selected_page == 'Pipelines' else ''}">Pipelines</a>
                <a href="?page=comparison" class="{'active' if selected_page == 'Comparison' else ''}">Comparison</a>
                <a href="?page=advanced_analysis" class="{'active' if selected_page == 'Advanced Analysis' else ''}">Advanced Analysis</a>
                <a href="?page=explainability" class="{'active' if selected_page == 'Explainability' else ''}">Explainability</a>
                <a href="?page=deploy" class="{'active' if selected_page == 'Deploy' else ''}">Deploy</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main application logic
def main():
    load_css()  # Load custom CSS for styling

    # Get the current page from query parameters
    query_params = st.experimental_get_query_params()
    selected_page = query_params.get("page", ["Home"])[0]

    # Render the navigation bar
    render_navigation(selected_page)

    # Render the content for the selected page
    if selected_page == "Home":
        st.title("Welcome to Awesome Scikit-Learn")
        st.write("Explore, learn, and build with Scikit-Learn.")
    elif selected_page == "Tutorial":
        st.title("Tutorial")
        st.write("Step-by-step tutorials to learn Scikit-Learn.")
    elif selected_page == "Explore":
        st.title("Explore")
        st.write("Upload and analyze datasets.")
    elif selected_page == "Train":
        st.title("Train")
        st.write("Train machine learning models interactively.")
    elif selected_page == "Evaluate":
        st.title("Evaluate")
        st.write("Evaluate model performance with metrics.")
    elif selected_page == "Visualize":
        st.title("Visualize")
        st.write("Visualize your data and model results.")
    elif selected_page == "Classification":
        st.title("Classification")
        st.write("Learn and apply classification algorithms.")
    elif selected_page == "Regression":
        st.title("Regression")
        st.write("Learn and apply regression algorithms.")
    elif selected_page == "Clustering":
        st.title("Clustering")
        st.write("Learn and apply clustering algorithms.")
    elif selected_page == "Pipelines":
        st.title("Pipelines")
        st.write("Create and manage machine learning pipelines.")
    elif selected_page == "Comparison":
        st.title("Comparison")
        st.write("Compare performance of multiple models.")
    elif selected_page == "Advanced Analysis":
        st.title("Advanced Analysis")
        st.write("Perform advanced data and model analysis.")
    elif selected_page == "Explainability":
        st.title("Explainability")
        st.write("Understand and explain model predictions.")
    elif selected_page == "Deploy":
        st.title("Deploy")
        st.write("Deploy your machine learning models.")
    else:
        st.title("404")
        st.write("Page not found.")

if __name__ == "__main__":
    main()
