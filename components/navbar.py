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
                <li><button id="home" class="nav-btn">Home</button></li>
                <li>
                    <button class="nav-btn">Models ▼</button>
                    <ul class="dropdown">
                        <li><button id="classification" class="nav-btn">Classification</button></li>
                        <li><button id="regression" class="nav-btn">Regression</button></li>
                        <li><button id="clustering" class="nav-btn">Clustering</button></li>
                    </ul>
                </li>
                <li>
                    <button class="nav-btn">Advanced ▼</button>
                    <ul class="dropdown">
                        <li><button id="pipelines" class="nav-btn">Pipelines</button></li>
                        <li><button id="comparison" class="nav-btn">Comparison</button></li>
                        <li><button id="advanced-analysis" class="nav-btn">Advanced Analysis</button></li>
                        <li><button id="explainability" class="nav-btn">Explainability</button></li>
                    </ul>
                </li>
                <li><button id="explore" class="nav-btn">Explore</button></li>
                <li><button id="train" class="nav-btn">Train</button></li>
                <li><button id="evaluate" class="nav-btn">Evaluate</button></li>
                <li><button id="deploy" class="nav-btn">Deploy</button></li>
                <li><button id="visualize" class="nav-btn">Visualize</button></li>
                <li><button id="tutorial" class="nav-btn">Tutorial</button></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main function to call navigation
def navigation_component():
    load_css()
    render_navigation()
