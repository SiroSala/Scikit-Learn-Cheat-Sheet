import streamlit as st

def load_sidebar_css():
    st.markdown(
        """
        <style>
        /* Sidebar container */
        .sidebar-container {
            height: 100%;
            width: 250px; /* Width of the sidebar */
            position: fixed;
            top: 0;
            left: 0;
            background-color: #2d3748; /* Dark background for the sidebar */
            padding-top: 20px;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
            overflow-x: hidden;
            z-index: 1000;
        }

        /* Sidebar navigation links */
        .sidebar a {
            padding: 10px 20px;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            color: #cbd5e0; /* Light text color */
            display: block;
            transition: all 0.3s ease;
        }

        /* Hover effects for links */
        .sidebar a:hover {
            background-color: #4a5568; /* Slightly lighter background on hover */
            color: #ffffff; /* White text on hover */
            border-left: 4px solid #63b3ed; /* Highlight with blue border */
        }

        /* Active link styling */
        .sidebar a.active {
            background-color: #1a202c; /* Highlight active link */
            color: #ffffff;
            border-left: 4px solid #63b3ed;
        }

        /* Sidebar header */
        .sidebar-header {
            font-size: 20px;
            font-weight: bold;
            color: #ffffff;
            text-align: center;
            margin-bottom: 20px;
        }

        /* Content container to adjust for sidebar */
        .content {
            margin-left: 260px; /* Same as the sidebar width + margin */
            padding: 20px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar():
    st.sidebar.title("Navigation")

def sidebar_component():
    load_sidebar_css()
    render_sidebar()
