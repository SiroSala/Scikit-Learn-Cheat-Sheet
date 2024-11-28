import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Awesome Scikit-Learn",
    page_icon="🔍",
    layout="wide"
)



def layout():
    st.title("Explore Data")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])
    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())
        
        # Visualize data
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
