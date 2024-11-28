import streamlit as st

def load_css():
    st.markdown(
        """
        <style>
        /* Main Navigation Container */
        .navigator-container {
            width: 100%;
            background-color: #ffffff; /* White background for modern look */
            padding: 15px 30px; /* Increased padding for better spacing */
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1); /* Soft shadow for elevation */
            display: flex;
            justify-content: center;
            align-items: center;
            position: sticky; /* Fix navigation bar at the top */
            top: 0;
            z-index: 1000;
        }

        /* Navigation List */
        .navigation {
            list-style: none;
            display: flex;
            margin: 0;
            padding: 0;
            gap: 25px; /* Increased gap for better link separation */
        }

        /* Navigation Links and Buttons */
        .navigation a,
        .navigation button {
            text-decoration: none;
            font-size: 16px; /* Standard font size */
            font-weight: 600; /* Bold text for emphasis */
            color: #374151; /* Neutral gray color for text */
            padding: 10px 15px; /* Comfortable padding for click/tap targets */
            border-radius: 6px; /* Slight rounding for modern design */
            background: none;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease; /* Smooth hover effect */
        }

        /* Hover Effects */
        .navigation a:hover,
        .navigation button:hover {
            background-color: #f3f4f6; /* Light gray hover background */
            color: #1f2937; /* Darker text color on hover */
        }

        /* Dropdown Menu */
        .dropdown {
            position: absolute;
            top: 110%; /* Dropdown below the link */
            left: 0;
            display: none;
            background-color: #ffffff; /* White background for dropdown */
            border: 1px solid #e5e7eb; /* Light gray border */
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Soft shadow */
            border-radius: 6px; /* Rounded dropdown corners */
            list-style: none;
            padding: 10px 0;
            z-index: 1000;
            width: 200px; /* Standard dropdown width */
        }

        .dropdown li {
            padding: 0;
        }

        .dropdown a {
            display: block;
            padding: 8px 15px; /* Padding inside dropdown items */
            font-size: 14px; /* Slightly smaller font for dropdown */
            font-weight: 500;
            color: #374151; /* Neutral gray for dropdown links */
            text-decoration: none;
            transition: all 0.2s ease; /* Smooth hover effect */
        }

        .dropdown a:hover {
            background-color: #f9fafb; /* Slightly lighter gray hover */
            color: #111827; /* Dark text on hover */
        }

        /* Show Dropdown on Hover */
        .navigation > li:hover .dropdown {
            display: block;
        }

        /* Responsiveness for Smaller Screens */
        @media (max-width: 768px) {
            .navigator-container {
                padding: 10px 20px;
            }
            .navigation {
                flex-wrap: wrap; /* Stack navigation items if necessary */
                gap: 15px; /* Smaller gap on smaller screens */
            }
            .navigation a,
            .navigation button {
                font-size: 14px; /* Smaller font for smaller screens */
                padding: 8px 10px; /* Adjust padding for smaller targets */
            }
            .dropdown {
                width: 180px; /* Slightly smaller dropdown width */
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
