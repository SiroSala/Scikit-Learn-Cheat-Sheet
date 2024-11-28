import streamlit as st
from streamlit_option_menu import option_menu

# Import custom pages
from pages import home, tutorial, explore, train, evaluate

# Set Streamlit page configuration
st.set_page_config(
    page_title="Awesome Scikit-Learn",
    layout="wide",
    initial_sidebar_state="collapsed"  # Ensure the sidebar starts collapsed
)

# Add horizontal navigation menu
with st.container():
    selected_page = option_menu(
        menu_title=None,  # No title for the menu
        options=["Home", "Tutorial", "Explore", "Train", "Evaluate"],  # Pages
        icons=["house", "book", "search", "tools", "bar-chart-line"],  # Page icons
        menu_icon="cast",  # Main menu icon
        default_index=0,  # Default selected page
        orientation="horizontal",  # Horizontal navigation
    )

# Dynamically load pages based on the selection
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






    # Footer with social media links
    st.markdown(f"""
        <div style="background-color: #FFFFFF; color: black; text-align: center; padding: 20px; margin-top: 50px; border-top: 2px solid #000000;">
            <p>Connect with me:</p>
            <div style="display: flex; justify-content: center; gap: 20px;">
                <a href="https://facebook.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" alt="Facebook" width="30" style="transition: transform 0.2s;">
                </a>
                <a href="https://instagram.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733558.png" alt="Instagram" width="30" style="transition: transform 0.2s;">
                </a>
                <a href="https://github.com/ahammadmejbah" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" alt="GitHub" width="30" style="transition: transform 0.2s;">
                </a>
                <a href="https://ahammadmejbah.com/" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/919/919827.png" alt="Portfolio" width="30" style="transition: transform 0.2s;">
                </a>
            </div>
            <br>
            Data Science Cheat Sheet v1.0 | Nov 2024 | <a href="https://ahammadmejbah.com/" style="color: #000000;">Mejbah Ahammad</a>
            <div class="card-footer">Mejbah Ahammad © 2024</div>
        </div>
    """, unsafe_allow_html=True)

    # Optional: Add some spacing at the bottom
    st.markdown("<br><br><br>", unsafe_allow_html=True)