import streamlit as st

# Custom CSS for the full-width navigation bar with dropdowns
def load_css():
    st.markdown(
        """
        <style>
        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #f8f9fa;
            padding: 10px 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            width: 100%;
        }
        .navbar-logo {
            display: flex;
            align-items: center;
        }
        .navbar-logo img {
            height: 50px;
            margin-right: 15px;
        }
        .navbar-title {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
        }
        .navbar-links {
            display: flex;
            justify-content: center;
            flex-grow: 1;
        }
        .navbar-links > div {
            position: relative;
            margin: 0 15px;
        }
        .navbar-links a {
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
            text-decoration: none;
            margin: 0 10px;
            transition: color 0.3s ease;
        }
        .navbar-links a:hover {
            color: #0056b3;
        }
        .dropdown {
            display: none;
            position: absolute;
            top: 35px;
            left: 0;
            background-color: #ffffff;
            border: 1px solid #ddd;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }
        .dropdown a {
            display: block;
            padding: 10px 15px;
            color: #007bff;
            text-decoration: none;
            font-size: 16px;
        }
        .dropdown a:hover {
            background-color: #f1f1f1;
        }
        .navbar-links > div:hover .dropdown {
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Render the full-width navigation bar
def render_navigation(selected_page):
    st.markdown(
        f"""
        <div class="navbar">
            <div class="navbar-logo">
                <img src="https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png" alt="Logo">
                <h2 class="navbar-title">Awesome Scikit-Learn</h2>
            </div>
            <div class="navbar-links">
                <a href="?page=home" class="{'active' if selected_page == 'Home' else ''}">Home</a>
                <a href="?page=tutorial" class="{'active' if selected_page == 'Tutorial' else ''}">Tutorial</a>
                <div>
                    <a href="#" class="nav-item">Models ▼</a>
                    <div class="dropdown">
                        <a href="?page=classification" class="{'active' if selected_page == 'Classification' else ''}">Classification</a>
                        <a href="?page=regression" class="{'active' if selected_page == 'Regression' else ''}">Regression</a>
                        <a href="?page=clustering" class="{'active' if selected_page == 'Clustering' else ''}">Clustering</a>
                    </div>
                </div>
                <div>
                    <a href="#" class="nav-item">Advanced ▼</a>
                    <div class="dropdown">
                        <a href="?page=pipelines" class="{'active' if selected_page == 'Pipelines' else ''}">Pipelines</a>
                        <a href="?page=comparison" class="{'active' if selected_page == 'Comparison' else ''}">Comparison</a>
                        <a href="?page=advanced_analysis" class="{'active' if selected_page == 'Advanced Analysis' else ''}">Advanced Analysis</a>
                    </div>
                </div>
                <a href="?page=evaluate" class="{'active' if selected_page == 'Evaluate' else ''}">Evaluate</a>
                <a href="?page=deploy" class="{'active' if selected_page == 'Deploy' else ''}">Deploy</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main application logic
def main():
    load_css()  # Load custom CSS

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
    elif selected_page == "Evaluate":
        st.title("Evaluate")
        st.write("Evaluate model performance with metrics.")
    elif selected_page == "Deploy":
        st.title("Deploy")
        st.write("Deploy your machine learning models.")
    else:
        st.title("404")
        st.write("Page not found.")

if __name__ == "__main__":
    main()
