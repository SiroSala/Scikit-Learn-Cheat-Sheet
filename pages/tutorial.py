import streamlit as st
import os

def load_markdown_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def main():
    # Define the path to the Markdown file
    markdown_path = "content/tutorial.md"
    
    # Load and display the content of the Markdown file
    if os.path.exists(markdown_path):
        tutorial_content = load_markdown_file(markdown_path)
        st.markdown(tutorial_content, unsafe_allow_html=True)
    else:
        st.error("Error: Tutorial content file not found.")

if __name__ == "__main__":
    main()
