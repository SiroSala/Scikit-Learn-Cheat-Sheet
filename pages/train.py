import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.set_page_config(
    page_title="Awesome Scikit-Learn",
    page_icon="🔍",
    layout="wide"
)


def layout():
    st.title("Train Models")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV only)", type=["csv"])
    if uploaded_file:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())
        
        # Train-Test Split
        st.subheader("Train-Test Split")
        test_size = st.slider("Select test size (%)", 10, 50, 20)
        train_data, test_data = train_test_split(df, test_size=test_size/100)
        st.write("Train Set Size:", train_data.shape)
        st.write("Test Set Size:", test_data.shape)
        
        # Model Training
        st.subheader("Train a Random Forest Classifier")
        if st.button("Train Model"):
            features = train_data.iloc[:, :-1]  # Assume last column is target
            target = train_data.iloc[:, -1]
            model = RandomForestClassifier()
            model.fit(features, target)
            st.success("Model trained successfully!")
