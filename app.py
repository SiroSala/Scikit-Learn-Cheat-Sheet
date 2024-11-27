import openai
import streamlit as st
import logging
from PIL import Image, ImageEnhance
import time
import json
import requests
import base64
from openai import OpenAI, OpenAIError

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 50
API_DOCS_URL = "https://scikit-learn.org/stable/documentation.html"

# Retrieve and validate API key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    st.error("Please add your OpenAI API key to the Streamlit secrets.toml file.")
    st.stop()

# Assign OpenAI API Key
openai.api_key = OPENAI_API_KEY
client = openai.OpenAI()

# Streamlit Page Configuration
st.set_page_config(
    page_title="SkLearnly - Your Comprehensive Scikit-learn Assistant",
    page_icon="https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/AdieLaine/SkLearnly",
        "Report a bug": "https://github.com/AdieLaine/SkLearnly",
        "About": """
            ## SkLearnly Scikit-learn Assistant
            ### Powered using GPT-4o-mini

            **GitHub**: https://github.com/AdieLaine/

            The AI Assistant named SkLearnly aims to provide the latest updates from Scikit-learn,
            generate code snippets for Scikit-learn models and utilities,
            and answer questions about Scikit-learn's latest features, issues, and more.
            SkLearnly has been trained on the latest Scikit-learn updates and documentation.
        """
    }
)

# Streamlit Title with Logo
def display_header():
    col1, col2 = st.columns([1, 3])
    with col1:
        logo_image_url = "https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png"
        img_base64_logo = img_to_base64(logo_image_url)
        if img_base64_logo:
            st.image(f"data:image/png;base64,{img_base64_logo}", width=100)
    with col2:
        st.title("SkLearnly - Your Comprehensive Scikit-learn Assistant")

# Function to convert image to base64
def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path)
            return base64.b64encode(response.content).decode()
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.error(f"Error converting image to base64: {str(e)}")
        return None

# Cache long-running tasks
@st.cache_data(show_spinner=False)
def long_running_task(duration):
    """
    Simulates a long-running operation.

    Parameters:
    - duration: int, duration of the task in seconds

    Returns:
    - str: Completion message
    """
    time.sleep(duration)
    return "Long-running operation completed."

# Load and enhance image
@st.cache_data(show_spinner=False)
def load_and_enhance_image(image_path, enhance=False):
    """
    Load and optionally enhance an image.

    Parameters:
    - image_path: str, path of the image
    - enhance: bool, whether to enhance the image or not

    Returns:
    - img: PIL.Image.Image, (enhanced) image
    """
    img = Image.open(image_path)
    if enhance:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
    return img

# Load Scikit-learn updates from JSON
@st.cache_data(show_spinner=False)
def load_scikit_learn_updates():
    """Load the latest Scikit-learn updates from a local JSON file."""
    try:
        with open("data/scikit_learn_updates.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading JSON: {str(e)}")
        return {}

# Get Scikit-learn API version
def get_scikit_learn_api_code_version():
    """
    Get the current Scikit-learn API code version from the Scikit-learn API documentation.

    Returns:
    - str: The current Scikit-learn API code version.
    """
    try:
        response = requests.get(API_DOCS_URL)
        if response.status_code == 200:
            return "1.3.2"
    except requests.exceptions.RequestException as e:
        logging.error(f"Error connecting to the Scikit-learn API documentation: {str(e)}")
    return None

# Display Scikit-learn updates
def display_scikit_learn_updates():
    """Display the latest updates of Scikit-learn."""
    st.header("📢 Latest Scikit-learn Updates")
    latest_updates = load_scikit_learn_updates()
    formatted_message = construct_formatted_message(latest_updates)
    st.markdown(formatted_message)

# Initialize conversation history
def initialize_conversation():
    """
    Initialize the conversation history with system and assistant messages.

    Returns:
    - list: Initialized conversation history.
    """
    assistant_message = "Hello! I am SkLearnly. How can I assist you with Scikit-learn today?"

    conversation_history = [
        {"role": "system", "content": "You are SkLearnly, a specialized AI assistant trained in Scikit-learn."},
        {"role": "system", "content": "SkLearnly is powered by the OpenAI GPT-4o-mini model, released on July 18, 2024."},
        {"role": "system", "content": "You are trained up to Scikit-learn Version 1.3.2, released on June 20, 2024."},
        {"role": "system", "content": "Refer to conversation history to provide context to your response."},
        {"role": "system", "content": "You were created by Mejbah Ahammad, an OpenAI Researcher."},
        {"role": "assistant", "content": assistant_message}
    ]
    return conversation_history

# Fetch latest update based on keyword
@st.cache_data(show_spinner=False)
def get_latest_update_from_json(keyword, latest_updates):
    """
    Fetch the latest Scikit-learn update based on a keyword.

    Parameters:
    - keyword (str): The keyword to search for in the Scikit-learn updates.
    - latest_updates (dict): The latest Scikit-learn updates data.

    Returns:
    - str: The latest update related to the keyword, or a message if no update is found.
    """
    for section in ["Highlights", "New Features", "Improvements", "Bug Fixes"]:
        for sub_key, sub_value in latest_updates.get(section, {}).items():
            for key, value in sub_value.items():
                if keyword.lower() in key.lower() or keyword.lower() in value.lower():
                    return f"**Section:** {section}\n**Sub-Category:** {sub_key}\n**{key}:** {value}"
    return "No updates found for the specified keyword."

# Construct formatted message for updates
def construct_formatted_message(latest_updates):
    """
    Construct formatted message for the latest updates.

    Parameters:
    - latest_updates (dict): The latest Scikit-learn updates data.

    Returns:
    - str: Formatted update messages.
    """
    formatted_message = []
    highlights = latest_updates.get("Highlights", {})
    version_info = highlights.get("Version 1.3.2", {})
    if version_info:
        description = version_info.get("Description", "No description available.")
        formatted_message.append(f"- **Version 1.3.2**: {description}")

    for category, updates in latest_updates.items():
        if category != "Highlights":
            formatted_message.append(f"**{category}**:")
            for sub_key, sub_values in updates.items():
                if sub_key != "Version 1.3.2":
                    description = sub_values.get("Description", "No description available.")
                    documentation = sub_values.get("Documentation", "No documentation available.")
                    formatted_message.append(f"- **{sub_key}**: {description}")
                    formatted_message.append(f"  - **Documentation**: {documentation}")
    return "\n".join(formatted_message)

# Handle chat submissions
@st.cache_data(show_spinner=False)
def on_chat_submit(chat_input, latest_updates):
    """
    Handle chat input submissions and interact with the OpenAI API.

    Parameters:
    - chat_input (str): The chat input from the user.
    - latest_updates (dict): The latest Scikit-learn updates fetched from a JSON file or API.

    Returns:
    - None: Updates the chat history in Streamlit's session state.
    """
    user_input = chat_input.strip()

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = initialize_conversation()

    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        model_engine = "gpt-4o-mini"
        assistant_reply = ""

        if "latest updates" in user_input.lower():
            assistant_reply = "Here are the latest highlights from Scikit-learn:\n"
            highlights = latest_updates.get("Highlights", {})
            if highlights:
                for version, info in highlights.items():
                    description = info.get("Description", "No description available.")
                    assistant_reply += f"- **{version}**: {description}\n"
            else:
                assistant_reply = "No highlights found."
        else:
            response = client.chat.completions.create(
                model=model_engine,
                messages=st.session_state.conversation_history
            )
            assistant_reply = response.choices[0].message.content

        st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except OpenAIError as e:
        logging.error(f"Error occurred: {e}")
        st.error(f"OpenAI Error: {str(e)}")

# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

# Additional Features Placeholder
def additional_features():
    """Include additional comprehensive features in the SkLearnly assistant."""
    pass  # Placeholder for additional features if needed

# Comprehensive SkLearnly Assistant with All Features
def full_comprehensive_sklearnly():
    import streamlit as st
    import openai
    from PIL import Image
    import json
    import logging
    import requests
    import base64

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Load OpenAI API key
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
    openai.api_key = OPENAI_API_KEY

    def load_conversation_history():
        """Load conversation history from session state."""
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        return st.session_state.conversation_history

    def update_conversation(role, content):
        """Update conversation history."""
        st.session_state.conversation_history.append({"role": role, "content": content})

    def generate_response(conversation):
        """Generate a response from OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=conversation
            )
            return response.choices[0].message.content
        except openai.OpenAIError as e:
            logging.error(f"OpenAI API Error: {e}")
            return "I'm sorry, I couldn't process that. Please try again later."

    def chat_interface():
        """Handle the chat interface."""
        st.header("🤖 Chat with SkLearnly")

        if 'conversation_history' not in st.session_state:
            initialize_conversation()

        conversation = load_conversation_history()

        user_input = st.text_input("You:", "")
        if st.button("Send"):
            if user_input:
                update_conversation("user", user_input)
                response = generate_response(conversation)
                update_conversation("assistant", response)
                st.experimental_rerun()

        # Display conversation
        for msg in conversation:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**SkLearnly:** {msg['content']}")

    def initialize_conversation():
        """Initialize the conversation history."""
        welcome_message = "Hello! I'm SkLearnly, your Scikit-learn assistant. How can I help you today?"
        update_conversation("assistant", welcome_message)

    def latest_updates_section():
        """Display the latest Scikit-learn updates."""
        st.header("📢 Latest Scikit-learn Updates")
        try:
            response = requests.get("https://api.scikit-learn.org/v1/updates")
            if response.status_code == 200:
                updates = response.json()
                for update in updates:
                    st.subheader(update.get("title", "No Title"))
                    st.write(update.get("description", "No Description"))
            else:
                st.error("Failed to fetch the latest updates.")
        except Exception as e:
            logging.error(f"Error fetching updates: {e}")
            st.error("An error occurred while fetching updates.")

    def code_generation_section():
        """Handle code generation based on user input."""
        st.header("💻 Generate Scikit-learn Code")
        code_prompt = st.text_area("Describe the Scikit-learn component or feature you need code for:")
        if st.button("Generate Code"):
            if code_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that generates code snippets."},
                    {"role": "user", "content": code_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    code = response.choices[0].message.content
                    st.code(code, language='python')
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to generate code. Please try again later.")
            else:
                st.warning("Please enter a description for code generation.")

    def troubleshooting_section():
        """Provide troubleshooting assistance."""
        st.header("🛠️ Troubleshooting")
        issue = st.text_area("Describe the issue you're facing with Scikit-learn:")
        if st.button("Get Help"):
            if issue:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that helps troubleshoot issues."},
                    {"role": "user", "content": issue}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    solution = response.choices[0].message.content
                    st.write(solution)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide a solution. Please try again later.")
            else:
                st.warning("Please describe the issue you need help with.")

    def code_explanation_section():
        """Explain provided Scikit-learn code."""
        st.header("📖 Code Explanation")
        code_snippet = st.text_area("Paste your Scikit-learn code here for explanation:")
        if st.button("Explain Code"):
            if code_snippet:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that explains code snippets."},
                    {"role": "user", "content": f"Explain the following Scikit-learn code:\n\n{code_snippet}"}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    explanation = response.choices[0].message.content
                    st.write(explanation)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to explain code. Please try again later.")
            else:
                st.warning("Please paste the code you want explained.")

    def project_analysis_section():
        """Analyze the user's Scikit-learn project."""
        st.header("📊 Project Analysis")
        project_description = st.text_area("Describe your Scikit-learn project:")
        if st.button("Analyze Project"):
            if project_description:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that analyzes and provides feedback on projects."},
                    {"role": "user", "content": project_description}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    analysis = response.choices[0].message.content
                    st.write(analysis)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to analyze project. Please try again later.")
            else:
                st.warning("Please describe your project for analysis.")

    def debug_assistance_section():
        """Assist in debugging Scikit-learn code."""
        st.header("🐞 Debug Assistance")
        debug_code = st.text_area("Paste your Scikit-learn code with errors:")
        if st.button("Debug Code"):
            if debug_code:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that helps debug code."},
                    {"role": "user", "content": f"Debug the following Scikit-learn code:\n\n{debug_code}"}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    debug_solution = response.choices[0].message.content
                    st.write(debug_solution)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to debug code. Please try again later.")
            else:
                st.warning("Please paste the code you need help debugging.")

    def interactive_visualization_section():
        """Generate interactive visualizations using Plotly."""
        st.header("📈 Interactive Visualizations")
        viz_prompt = st.text_area("Describe the visualization you need:")
        if st.button("Generate Visualization"):
            if viz_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that generates interactive Plotly visualizations."},
                    {"role": "user", "content": viz_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    viz_code = response.choices[0].message.content
                    st.code(viz_code, language='python')
                    exec(viz_code, globals())
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to generate visualization. Please try again later.")
                except Exception as e:
                    logging.error(f"Error executing visualization code: {e}")
                    st.error("An error occurred while generating the visualization.")
            else:
                st.warning("Please describe the visualization you need.")

    def data_cleaning_section():
        """Provide data cleaning techniques."""
        st.header("🧹 Data Cleaning Techniques")
        cleaning_prompt = st.text_area("Describe the data cleaning task you need assistance with:")
        if st.button("Get Cleaning Tips"):
            if cleaning_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that provides data cleaning techniques using Pandas."},
                    {"role": "user", "content": cleaning_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    cleaning_tips = response.choices[0].message.content
                    st.write(cleaning_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide cleaning tips. Please try again later.")
            else:
                st.warning("Please describe the data cleaning task you need assistance with.")

    def feature_engineering_section():
        """Assist with feature engineering."""
        st.header("🔧 Feature Engineering")
        feature_prompt = st.text_area("Describe the feature engineering task you need help with:")
        if st.button("Get Feature Suggestions"):
            if feature_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that helps with feature engineering using Pandas."},
                    {"role": "user", "content": feature_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    feature_suggestions = response.choices[0].message.content
                    st.write(feature_suggestions)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide feature suggestions. Please try again later.")
            else:
                st.warning("Please describe the feature engineering task you need help with.")

    def handle_large_datasets_section():
        """Provide techniques for handling large datasets."""
        st.header("📂 Handling Large Datasets")
        dataset_prompt = st.text_area("Describe the large dataset handling task you need assistance with:")
        if st.button("Get Handling Techniques"):
            if dataset_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that provides techniques for handling large datasets using Pandas."},
                    {"role": "user", "content": dataset_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    handling_tips = response.choices[0].message.content
                    st.write(handling_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide handling techniques. Please try again later.")
            else:
                st.warning("Please describe the large dataset handling task you need assistance with.")

    def performance_optimization_section():
        """Provide performance optimization tips."""
        st.header("⚡ Performance Optimization")
        optimization_prompt = st.text_area("Describe the performance optimization task you need assistance with:")
        if st.button("Get Optimization Tips"):
            if optimization_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that provides performance optimization tips using Pandas."},
                    {"role": "user", "content": optimization_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    optimization_tips = response.choices[0].message.content
                    st.write(optimization_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide optimization tips. Please try again later.")
            else:
                st.warning("Please describe the performance optimization task you need assistance with.")

    def integration_section():
        """Provide integration tips with other libraries."""
        st.header("🔗 Integration with Other Libraries")
        integration_prompt = st.text_area("Describe the integration task you need assistance with:")
        if st.button("Get Integration Tips"):
            if integration_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that provides integration tips with other libraries."},
                    {"role": "user", "content": integration_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    integration_tips = response.choices[0].message.content
                    st.write(integration_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide integration tips. Please try again later.")
            else:
                st.warning("Please describe the integration task you need assistance with.")

    def machine_learning_pipelines_section():
        """Assist with building machine learning pipelines."""
        st.header("🤖 Machine Learning Pipelines")
        ml_prompt = st.text_area("Describe the machine learning pipeline task you need assistance with:")
        if st.button("Get ML Pipeline Tips"):
            if ml_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that helps build machine learning pipelines using Pandas and Scikit-learn."},
                    {"role": "user", "content": ml_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    ml_tips = response.choices[0].message.content
                    st.write(ml_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide ML pipeline tips. Please try again later.")
            else:
                st.warning("Please describe the machine learning pipeline task you need assistance with.")

    def export_sharing_section():
        """Assist with exporting and sharing results."""
        st.header("📤 Exporting and Sharing Results")
        export_prompt = st.text_area("Describe how you want to export or share your results:")
        if st.button("Get Export Tips"):
            if export_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that helps with exporting and sharing results."},
                    {"role": "user", "content": export_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    export_tips = response.choices[0].message.content
                    st.write(export_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide export tips. Please try again later.")
            else:
                st.warning("Please describe how you want to export or share your results.")

    def working_with_dates_section():
        """Provide advanced datetime operations."""
        st.header("📅 Working with Dates and Times")
        dates_prompt = st.text_area("Describe the datetime task you need assistance with:")
        if st.button("Get DateTime Tips"):
            if dates_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that provides advanced datetime operations using Pandas."},
                    {"role": "user", "content": dates_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    datetime_tips = response.choices[0].message.content
                    st.write(datetime_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide datetime tips. Please try again later.")
            else:
                st.warning("Please describe the datetime task you need assistance with.")

    def handling_duplicates_section():
        """Assist with handling duplicate data."""
        st.header("🔄 Handling Duplicates")
        duplicates_prompt = st.text_area("Describe the duplicate data handling task you need assistance with:")
        if st.button("Get Duplicate Handling Tips"):
            if duplicates_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that helps handle duplicate data using Pandas."},
                    {"role": "user", "content": duplicates_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    duplicate_tips = response.choices[0].message.content
                    st.write(duplicate_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide duplicate handling tips. Please try again later.")
            else:
                st.warning("Please describe the duplicate data handling task you need assistance with.")

    def data_normalization_section():
        """Provide data normalization and scaling tips."""
        st.header("📏 Data Normalization and Scaling")
        normalization_prompt = st.text_area("Describe the data normalization or scaling task you need assistance with:")
        if st.button("Get Normalization Tips"):
            if normalization_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that provides data normalization and scaling tips using Scikit-learn."},
                    {"role": "user", "content": normalization_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    normalization_tips = response.choices[0].message.content
                    st.write(normalization_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide normalization tips. Please try again later.")
            else:
                st.warning("Please describe the data normalization or scaling task you need assistance with.")

    def text_data_processing_section():
        """Assist with handling and analyzing textual data."""
        st.header("📝 Text Data Processing")
        text_prompt = st.text_area("Describe the text data processing task you need assistance with:")
        if st.button("Get Text Processing Tips"):
            if text_prompt:
                conversation = [
                    {"role": "system", "content": "You are SkLearnly, a Scikit-learn assistant that helps with text data processing using Pandas."},
                    {"role": "user", "content": text_prompt}
                ]
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=conversation
                    )
                    text_tips = response.choices[0].message.content
                    st.write(text_tips)
                except openai.OpenAIError as e:
                    logging.error(f"OpenAI API Error: {e}")
                    st.error("Failed to provide text processing tips. Please try again later.")
            else:
                st.warning("Please describe the text data processing task you need assistance with.")

    def about_section():
        """Display information about SkLearnly."""
        st.header("ℹ️ About SkLearnly")
        st.markdown("""
        **SkLearnly** is an intelligent assistant designed to help you with all your Scikit-learn needs. Whether you're looking for the latest updates, need help generating code snippets, require assistance with troubleshooting, want to analyze your projects, or need advanced data handling techniques, SkLearnly is here to assist you.

        - **Latest Updates:** Stay informed with the most recent features and changes in Scikit-learn.
        - **Code Generation:** Quickly generate Scikit-learn code snippets for your projects.
        - **Troubleshooting:** Get help with any issues or errors you encounter.
        - **Code Explanation:** Understand your Scikit-learn code with detailed explanations.
        - **Project Analysis:** Receive insights and recommendations on your Scikit-learn projects.
        - **Debug Assistance:** Troubleshoot and fix errors in your Scikit-learn code.
        - **Interactive Visualizations:** Generate interactive Plotly visualizations for your data.
        - **Data Cleaning:** Learn techniques to clean and prepare your data using Pandas.
        - **Feature Engineering:** Get suggestions for creating new features from your data.
        - **Handling Large Datasets:** Discover methods to efficiently manage large datasets.
        - **Performance Optimization:** Optimize your Scikit-learn models for better performance.
        - **Integration with Other Libraries:** Learn how to integrate Scikit-learn with other Python libraries.
        - **Machine Learning Pipelines:** Build robust machine learning pipelines using Scikit-learn.
        - **Exporting and Sharing Results:** Learn how to export and share your Scikit-learn model results.
        - **Working with Dates and Times:** Perform advanced datetime operations in your data.
        - **Handling Duplicates:** Manage and remove duplicate data effectively.
        - **Data Normalization and Scaling:** Prepare your data for machine learning models.
        - **Text Data Processing:** Handle and analyze textual data within your Scikit-learn projects.
        """)
        st.image("https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png", width=200)

    def main_full():
        """Main function to run the comprehensive SkLearnly assistant."""
        display_header()

        menu = [
            "Chat",
            "Latest Updates",
            "Generate Code",
            "Explain Code",
            "Analyze Project",
            "Debug Code",
            "Interactive Visualization",
            "Data Cleaning",
            "Feature Engineering",
            "Handle Large Datasets",
            "Performance Optimization",
            "Integration",
            "Machine Learning Pipelines",
            "Exporting Results",
            "Text Data Processing",
            "About"
        ]
        choice = st.sidebar.selectbox("Menu", menu)

        # Use Tabs for better layout
        if choice == "Chat":
            chat_interface()
        elif choice == "Latest Updates":
            latest_updates_section()
        elif choice == "Generate Code":
            code_generation_section()
        elif choice == "Explain Code":
            code_explanation_section()
        elif choice == "Analyze Project":
            project_analysis_section()
        elif choice == "Debug Code":
            debug_assistance_section()
        elif choice == "Interactive Visualization":
            interactive_visualization_section()
        elif choice == "Data Cleaning":
            data_cleaning_section()
        elif choice == "Feature Engineering":
            feature_engineering_section()
        elif choice == "Handle Large Datasets":
            handle_large_datasets_section()
        elif choice == "Performance Optimization":
            performance_optimization_section()
        elif choice == "Integration":
            integration_section()
        elif choice == "Machine Learning Pipelines":
            machine_learning_pipelines_section()
        elif choice == "Exporting Results":
            export_sharing_section()
        elif choice == "Text Data Processing":
            text_data_processing_section()
        elif choice == "About":
            about_section()

    if __name__ == "__main__":
        initialize_session_state()
        main_full()
