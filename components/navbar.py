import streamlit as st

def load_css():
    st.markdown(
        """
        <style>
        /* Main Navigation Bar */
        .navigator-container {
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 10px 0;
        }

        /* Navigation List */
        .navigation {
            list-style: none;
            display: flex;
            gap: 20px;
            margin: 0;
            padding: 0;
        }

        /* Navigation Links and Buttons */
        .navigation a {
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            color: #374151;
            padding: 8px 12px;
            border: none;
            background: none;
            border-bottom: 2px solid transparent;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        /* Hover Effects */
        .navigation a:hover {
            color: #1f2937;
            border-bottom: 2px solid #1f2937;
        }

        /* Dropdown Menu */
        .dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            display: none;
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 6px;
            list-style: none;
            padding: 5px 0;
            z-index: 1000;
            min-width: 150px;
        }

        .dropdown li {
            padding: 0;
        }

        .dropdown a {
            display: block;
            padding: 8px 15px;
            font-size: 14px;
            font-weight: 500;
            color: #374151;
            text-decoration: none;
            text-align: left;
        }

        /* Hover Effects for Dropdown */
        .dropdown a:hover {
            background-color: #f9fafb;
            color: #111827;
        }

        /* Show Dropdown on Hover */
        .navigation > li:hover .dropdown {
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_navigation():
    # Build the navigation with Streamlit query parameters
    st.markdown(
        """
        <div class="navigator-container">
            <ul class="navigation">
                <li><a href="?page=home">Home</a></li>
                <li>
                    <a href="#">Models ▼</a>
                    <ul class="dropdown">
                        <li><a href="?page=classification">Classification</a></li>
                        <li><a href="?page=regression">Regression</a></li>
                        <li><a href="?page=clustering">Clustering</a></li>
                    </ul>
                </li>
                <li>
                    <a href="#">Advanced ▼</a>
                    <ul class="dropdown">
                        <li><a href="?page=pipelines">Pipelines</a></li>
                        <li><a href="?page=comparison">Comparison</a></li>
                        <li><a href="?page=advanced-analysis">Advanced Analysis</a></li>
                        <li><a href="?page=explainability">Explainability</a></li>
                    </ul>
                </li>
                <li><a href="?page=explore">Explore</a></li>
                <li><a href="?page=train">Train</a></li>
                <li><a href="?page=evaluate">Evaluate</a></li>
                <li><a href="?page=deploy">Deploy</a></li>
                <li><a href="?page=visualize">Visualize</a></li>
                <li><a href="?page=tutorial">Tutorial</a></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main function to call navigation
def navigation_component():
    load_css()
    render_navigation()
