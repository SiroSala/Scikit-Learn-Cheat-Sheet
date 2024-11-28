import streamlit as st

def main():
    st.sidebar.image("static/images/logo.png", use_column_width=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Tutorial", "Explore", "Train", "Evaluate", "Visualize"])

    if page == "Home":
        st.title("Welcome to Awesome Scikit-Learn!")
        st.write("This application helps you learn and explore Scikit-Learn.")
    # Add additional pages logic here...

if __name__ == "__main__":
    main()
