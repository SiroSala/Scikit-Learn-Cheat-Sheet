import streamlit as st

def load_css():
    st.markdown(
        """
        <style>
        /* Main Navigation Container */
        .navigator-container {
            position: relative;
            width: 100%;
            background-color: #f8f9fa;
            padding: 10px 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .navigation {
            list-style: none;
            display: flex;
            margin: 0;
            padding: 0;
            gap: 20px;
        }
        .navigation > li {
            position: relative;
        }
        .navigation a,
        .navigation button {
            text-decoration: none;
            font-size: 16px;
            font-weight: 600;
            color: #4a5568;
            padding: 8px 12px;
            border-radius: 5px;
            background: none;
            border: none;
            cursor: pointer;
        }
        .navigation a:hover,
        .navigation button:hover {
            background-color: #edf2f7;
            color: #2d3748;
        }
        .dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            display: none;
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 5px;
            list-style: none;
            padding: 5px 0;
            z-index: 10;
        }
        .dropdown li {
            padding: 0;
        }
        .dropdown a {
            display: block;
            padding: 8px 15px;
            font-size: 14px;
            font-weight: 500;
            color: #4a5568;
            text-decoration: none;
        }
        .dropdown a:hover {
            background-color: #e2e8f0;
            color: #2d3748;
        }
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
                <li><a href="/home">Home</a></li>
                <li>
                    <button>Models ▼</button>
                    <ul class="dropdown">
                        <li><a href="/classification">Classification</a></li>
                        <li><a href="/regression">Regression</a></li>
                        <li><a href="/clustering">Clustering</a></li>
                    </ul>
                </li>
                <li>
                    <button>Advanced ▼</button>
                    <ul class="dropdown">
                        <li><a href="/pipelines">Pipelines</a></li>
                        <li><a href="/comparison">Comparison</a></li>
                        <li><a href="/advanced-analysis">Advanced Analysis</a></li>
                        <li><a href="/explainability">Explainability</a></li>
                    </ul>
                </li>
                <li><a href="/explore">Explore</a></li>
                <li><a href="/train">Train</a></li>
                <li><a href="/evaluate">Evaluate</a></li>
                <li><a href="/deploy">Deploy</a></li>
                <li><a href="/visualize">Visualize</a></li>
                <li><a href="/tutorial">Tutorial</a></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Main function to call navigation
def navigation_component():
    load_css()
    render_navigation()
