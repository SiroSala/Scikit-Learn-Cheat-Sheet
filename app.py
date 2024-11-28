import streamlit as st

# Permanently disable the default Streamlit sidebar
st.set_page_config(
    page_title="Custom App",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapses the sidebar by default
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,
    }
)
