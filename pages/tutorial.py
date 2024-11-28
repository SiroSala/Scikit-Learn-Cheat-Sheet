import streamlit as st
import os

def load_markdown_file(path):
    """Utility function to read a Markdown file and return its contents."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def layout():
    """Loads and displays the content of the Markdown file within the Streamlit app."""
    # Define the path to the Markdown file
    markdown_path = "content/tutorial.md"
    
    # Check if the file exists and display it; otherwise, show an error message
    if os.path.exists(markdown_path):
        tutorial_content = load_markdown_file(markdown_path)
        st.markdown(tutorial_content, unsafe_allow_html=True)
    else:
        st.error("Error: Tutorial content file not found.")
