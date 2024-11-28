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
            padding: 10px 0; /* Adjust spacing around navigation */
            box-shadow: none; /* Remove background shadow */
        }

        /* Navigation List */
        .navigation {
            list-style: none;
            display: flex;
            gap: 20px; /* Space between navigation items */
            margin: 0;
            padding: 0;
        }

        /* Navigation Links and Buttons */
        .navigation a,
        .navigation button {
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            color: #374151; /* Neutral gray for modern look */
            padding: 8px 12px;
            border: none;
            background: none;
            border-bottom: 2px solid transparent; /* Indicator for active state */
            cursor: pointer;
            transition: all 0.3s ease;
        }

        /* Hover Effects */
        .navigation a:hover,
        .navigation button:hover {
            color: #1f2937; /* Darker gray */
            border-bottom: 2px solid #1f2937; /* Bottom indicator on hover */
        }

        /* Dropdown Menu */
        .dropdown {
            position: absolute;
            top: 100%; /* Align dropdown below parent */
            left: 0;
            display: none;
            background-color: #ffffff; /* Dropdown background */
            border: 1px solid #e5e7eb; /* Light gray border */
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Soft shadow */
            border-radius: 6px; /* Rounded corners */
            list-style: none;
            padding: 5px 0;
            z-index: 1000;
            min-width: 150px; /* Standard dropdown width */
        }

        .dropdown li {
            padding: 0;
        }

        .dropdown a {
            display: block;
            padding: 8px 15px; /* Inner padding for dropdown items */
            font-size: 14px; /* Slightly smaller font for dropdown */
            font-weight: 500;
            color: #374151;
            text-decoration: none;
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

        /* Ensure Navigation is Standardized */
        @media (max-width: 768px) {
            .navigator-container {
                flex-wrap: wrap; /* Allow items to wrap for smaller screens */
                justify-content: flex-start;
            }
            .navigation {
                gap: 10px; /* Reduce spacing on smaller screens */
            }
            .navigation a,
            .navigation button {
                font-size: 14px;
                padding: 6px 10px;
            }
            .dropdown {
                min-width: 120px;
            }
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
