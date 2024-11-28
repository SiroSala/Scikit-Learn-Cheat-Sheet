import streamlit as st
import os

# Function to read the tutorial content from a markdown file
def load_tutorial(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    return "### Content Not Found\nThe tutorial content could not be loaded."

def layout():
    # Load tutorial content from the markdown file
    tutorial_content = load_tutorial("content/tutorial.md")

    # Render the page
    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #4CAF50; font-family: Georgia, serif;">📖 Tutorial</h1>
            <p style="font-family: Georgia, serif; font-size: 16px;">
                A comprehensive guide to get started with the Data Science Cheat Sheet.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display the tutorial content
    st.markdown(tutorial_content, unsafe_allow_html=True)

    # Optional: Add a feedback section
    st.markdown("---")
    st.markdown(
        """
        <h3 style="text-align: center; font-family: Georgia, serif;">💡 Have Feedback?</h3>
        <p style="text-align: center; font-family: Georgia, serif;">
            If you have suggestions or want to report an issue, please <a href="https://github.com/ahammadmejbah/Data-Science-Cheat-Sheet/issues" target="_blank">open an issue</a> on GitHub.
        </p>
        """,
        unsafe_allow_html=True,
    )
