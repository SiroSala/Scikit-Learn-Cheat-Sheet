import streamlit as st
from components.navbar import navigation_component

def main():
    # Call the navigation component
    navigation_component()

    # Page Content
    st.title("Welcome to Scikit-Learn Navigator")
    st.write("Choose a section from the navigation bar.")

if __name__ == "__main__":
    main()
