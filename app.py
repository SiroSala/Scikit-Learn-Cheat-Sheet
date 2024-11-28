import streamlit as st
from streamlit_option_menu import option_menu

# Import custom pages
from pages import home, tutorial, explore, train, evaluate

# Set Streamlit page configuration
st.set_page_config(page_title="Awesome Scikit-Learn", layout="wide")

# Add horizontal navigation
with st.container():
    selected_page = option_menu(
        menu_title=None,  # Title for the menu (None means no title)
        options=["Home", "Tutorial", "Explore", "Train", "Evaluate"],  # Menu options
        icons=["house", "book", "search", "tools", "bar-chart-line"],  # Icons for each page
        menu_icon="cast",  # Icon for the menu
        default_index=0,  # Default active menu option
        orientation="horizontal",  # Horizontal navigation
    )

# Dynamically load the selected page
if selected_page == "Home":
    home.layout()
elif selected_page == "Tutorial":
    tutorial.layout()
elif selected_page == "Explore":
    explore.layout()
elif selected_page == "Train":
    train.layout()
elif selected_page == "Evaluate":
    evaluate.layout()
