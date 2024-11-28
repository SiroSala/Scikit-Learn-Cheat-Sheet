import streamlit as st
import os

def load_markdown_file(path):
    """Utility function to read a Markdown file and return its contents."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def pagination_buttons():
    """Create centered pagination buttons."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state['selected_topic_index'] > 0:
            if st.button('← Previous'):
                st.session_state['selected_topic_index'] -= 1
                st.experimental_rerun()
    with col3:
        if st.session_state['selected_topic_index'] < len(topics) - 1:
            if st.button('Next →'):
                st.session_state['selected_topic_index'] += 1
                st.experimental_rerun()

def layout():
    """Loads and displays the content of the Markdown file within the Streamlit app."""
    # Define the base path to the Markdown files directory
    base_path = "content/90days"
    
    # Initialize or adjust the session state for topic index
    if 'selected_topic_index' not in st.session_state:
        st.session_state['selected_topic_index'] = 0  # Start at the first file

    # Generate list of topics based on file naming convention
    global topics
    topics = [f"Day{i}_Topic.md" for i in range(1, 91)]  # Adjust range as needed for your files

    # Display top pagination buttons
    pagination_buttons()

    # Load and display the markdown after potential state change
    markdown_path = os.path.join(base_path, topics[st.session_state['selected_topic_index']])
    if os.path.exists(markdown_path):
        tutorial_content = load_markdown_file(markdown_path)
        st.markdown(tutorial_content, unsafe_allow_html=True)
    else:
        st.error("Error: Tutorial content file not found.")

    # Display bottom pagination buttons
    pagination_buttons()

