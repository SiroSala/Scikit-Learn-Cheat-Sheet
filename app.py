import streamlit as st
from streamlit_option_menu import option_menu
from components.footer import render_footer
# Import custom pages
from pages import home, tutorial, explore, train, evaluate

# Set Streamlit page configuration
st.set_page_config(
    page_title="Awesome Scikit-Learn",
    layout="wide",
    initial_sidebar_state="collapsed"  # Ensure the sidebar starts collapsed
)

# Add horizontal navigation menu
with st.container():
    selected_page = option_menu(
        menu_title=None,  # No title for the menu
        options=["Home", "Tutorial", "Explore", "Train", "Evaluate"],  # Pages
        icons=["house", "book", "search", "tools", "bar-chart-line"],  # Page icons
        menu_icon="cast",  # Main menu icon
        default_index=0,  # Default selected page
        orientation="horizontal",  # Horizontal navigation
    )

# Dynamically load pages based on the selection
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




# Footer with social media links

render_footer()
