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
    
    # Generate list of topics based on file naming convention
    topics = [f"Day{i}_Topic.md" for i in range(1, 91)]  # Adjust range as needed for your files

    # Dropdown to select a topic
    selected_topic_index = st.selectbox(
        "Select a topic",
        range(len(topics)),
        index=0,  # Default to the first file
        format_func=lambda x: topics[x].replace('_', ' ').replace('.md', '')  # Clean up file names for display
    )

    # Define the path to the selected Markdown file
    markdown_path = os.path.join(base_path, topics[selected_topic_index])
    
    # Check if the file exists and display it; otherwise, show an error message
    if os.path.exists(markdown_path):
        tutorial_content = load_markdown_file(markdown_path)
        st.markdown(tutorial_content, unsafe_allow_html=True)
    else:
        st.error("Error: Tutorial content file not found.")
    
    # Pagination for previous and next topics
    col1, col2 = st.columns([1, 1])
    if selected_topic_index > 0:  # There is a previous topic
        with col1:
            if st.button("Previous"):
                # Go to the previous topic
                st.experimental_rerun()
                st.session_state['selected_topic_index'] = selected_topic_index - 1

    if selected_topic_index < len(topics) - 1:  # There is a next topic
        with col2:
            if st.button("Next"):
                # Go to the next topic
                st.experimental_rerun()
                st.session_state['selected_topic_index'] = selected_topic_index + 1
