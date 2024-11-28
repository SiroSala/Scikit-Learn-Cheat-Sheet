import streamlit as st
import os

def load_markdown_file(path):
    """Utility function to read a Markdown file and return its contents."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def layout():
    """Loads and displays the content of the Markdown file within the Streamlit app."""
    # Define the base path to the Markdown files directory
    base_path = "content/90days"
    
    # Initialize the session state for topic index if it's not already set
    if 'selected_topic_index' not in st.session_state:
        st.session_state['selected_topic_index'] = 0

    # Generate list of topics based on file naming convention
    topics = [f"Day{i}_Topic.md" for i in range(1, 91)]  # Adjust range as needed for your files

    # Define the path to the selected Markdown file
    markdown_path = os.path.join(base_path, topics[st.session_state['selected_topic_index']])
    
    # Check if the file exists and display it; otherwise, show an error message
    if os.path.exists(markdown_path):
        tutorial_content = load_markdown_file(markdown_path)
        st.markdown(tutorial_content, unsafe_allow_html=True)
    else:
        st.error("Error: Tutorial content file not found.")

    # Create buttons for navigation between topics
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state['selected_topic_index'] > 0:
            if st.button('Previous'):
                st.session_state['selected_topic_index'] -= 1
                st.experimental_rerun()

    with col2:
        if st.session_state['selected_topic_index'] < len(topics) - 1:
            if st.button('Next'):
                st.session_state['selected_topic_index'] += 1
                st.experimental_rerun()
