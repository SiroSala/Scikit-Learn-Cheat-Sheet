import streamlit as st
from streamlit_lottie import st_lottie
from footer import display_footer  # Import footer module
from pages import home, tutorial, explore, train, evaluate  # Import your pages
import requests

# Ensure this is the first command
st.set_page_config(
    page_title="Awesome Scikit-Learn",
    page_icon="🔍",
    layout="wide"
)

# Function to load Lottie animations from a URL (optional for app-wide animations)
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Add horizontal navigation menu
selected_page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Tutorial", "Explore", "Train", "Evaluate"],
)

# Render the selected page
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

# Add Footer
display_footer()
