import streamlit as st
import os

def load_markdown_file(path):
    """Utility function to read a Markdown file and return its contents."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def pagination_buttons(position):
    """Create centered pagination buttons with unique keys based on position (top or bottom)."""
    # Apply custom styles to the buttons
    button_style = """
    <style>
        .stButton>button {
            border: 2px solid #4CAF50;
            color: white;
            background-color: #4CAF50;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            transition-duration: 0.4s;
            cursor: pointer;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: white;
            color: black;
        }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.session_state['selected_topic_index'] > 0:
            if st.button('← Previous', key=f'prev_{position}'):
                st.session_state['selected_topic_index'] -= 1
                # Force rerun by manipulating session state
                st.session_state['rerun'] = not st.session_state.get('rerun', False)
    with col3:
        if st.session_state['selected_topic_index'] < len(topics) - 1:
            if st.button('Next →', key=f'next_{position}'):
                st.session_state['selected_topic_index'] += 1
                # Force rerun by manipulating session state
                st.session_state['rerun'] = not st.session_state.get('rerun', False)

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
    pagination_buttons('top')

    # Load and display the markdown after potential state change
    markdown_path = os.path.join(base_path, topics[st.session_state['selected_topic_index']])
    if os.path.exists(markdown_path):
        tutorial_content = load_markdown_file(markdown_path)
        st.markdown(tutorial_content, unsafe_allow_html=True)
    else:
        st.error("Error: Tutorial content file not found.")

    # Display bottom pagination buttons
    pagination_buttons('bottom')

