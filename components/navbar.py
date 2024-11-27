import streamlit as st

def render():
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="static/images/logo.png" alt="Logo" style="height: 50px; margin-right: 10px;">
            <h1>Awesome Scikit-Learn</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
