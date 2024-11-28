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
        .navigation a,
        .navigation button {
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
        .navigation a:hover,
        .navigation button:hover {
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

        .dropdown button {
            display: block;
            padding: 8px 15px;
            font-size: 14px;
            font-weight: 500;
            color: #374151;
            text-decoration: none;
            background: none;
            border: none;
            cursor: pointer;
            text-align: left;
        }

        /* Hover Effects for Dropdown */
        .dropdown button:hover {
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
    st.markdown(
        """
        <div class="navigator-container">
            <ul class="navigation">
                <li><button onclick="setPage('home')">Home</button></li>
                <li>
                    <button>Models ▼</button>
                    <ul class="dropdown">
                        <li><button onclick="setPage('classification')">Classification</button></li>
                        <li><button onclick="setPage('regression')">Regression</button></li>
                        <li><button onclick="setPage('clustering')">Clustering</button></li>
                    </ul>
                </li>
                <li>
                    <button>Advanced ▼</button>
                    <ul class="dropdown">
                        <li><button onclick="setPage('pipelines')">Pipelines</button></li>
                        <li><button onclick="setPage('comparison')">Comparison</button></li>
                        <li><button onclick="setPage('advanced-analysis')">Advanced Analysis</button></li>
                        <li><button onclick="setPage('explainability')">Explainability</button></li>
                    </ul>
                </li>
                <li><button onclick="setPage('explore')">Explore</button></li>
                <li><button onclick="setPage('train')">Train</button></li>
                <li><button onclick="setPage('evaluate')">Evaluate</button></li>
                <li><button onclick="setPage('deploy')">Deploy</button></li>
                <li><button onclick="setPage('visualize')">Visualize</button></li>
                <li><button onclick="setPage('tutorial')">Tutorial</button></li>
            </ul>
        </div>
        <script>
            function setPage(page) {
                const queryString = new URLSearchParams(window.location.search);
                queryString.set("page", page);
                window.location.search = queryString.toString();
            }
        </script>
        """,
        unsafe_allow_html=True,
    )

# Main function to call navigation
def navigation_component():
    load_css()
    render_navigation()
