import streamlit as st
import base64
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Import datasets and models from scikit-learn
from sklearn import datasets
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                     GridSearchCV, RandomizedSearchCV, 
                                     StratifiedKFold, learning_curve, 
                                     TimeSeriesSplit)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                   OneHotEncoder, LabelEncoder, PolynomialFeatures, 
                                   Normalizer, Binarizer, KBinsDiscretizer, 
                                   OrdinalEncoder)

# Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import (SelectKBest, chi2, RFE, RFECV, 
                                       SelectFromModel, VarianceThreshold)
from sklearn.linear_model import (LinearRegression, LogisticRegression, 
                                  Ridge, Lasso, ElasticNet, SGDRegressor, 
                                  SGDClassifier, Perceptron, PassiveAggressiveClassifier)
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                              AdaBoostClassifier, AdaBoostRegressor, 
                              GradientBoostingClassifier, GradientBoostingRegressor, 
                              VotingClassifier, VotingRegressor, BaggingClassifier, 
                              BaggingRegressor, StackingClassifier, StackingRegressor, 
                              IsolationForest, ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering, 
                             MeanShift, SpectralClustering, OPTICS, Birch)
from sklearn.decomposition import (PCA, TruncatedSVD, NMF, LatentDirichletAllocation,
                                   FastICA, KernelPCA, SparsePCA)
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding, LocallyLinearEmbedding
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             mean_squared_error, r2_score, roc_curve, auc, 
                             silhouette_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, mean_absolute_error, 
                             adjusted_rand_score, adjusted_mutual_info_score, 
                             homogeneity_score, completeness_score, v_measure_score, 
                             calinski_harabasz_score, davies_bouldin_score, 
                             mean_squared_log_error, explained_variance_score, 
                             mean_absolute_percentage_error, brier_score_loss)
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin
from sklearn.datasets import (make_regression, make_classification, make_blobs, 
                              load_wine, load_diabetes, 
                              load_breast_cancer, load_digits, load_iris)
import joblib

# Set page configuration
st.set_page_config(
    page_title='🤖 Scikit-Learn Ultimate Cheat Sheet by Mejbah Ahammad',
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def img_to_bytes(img_url):
    try:
        response = requests.get(img_url)
        img_bytes = response.content
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except:
        return ''

def main():
    st.markdown(
        """
        <style>
        .css-1v3fvcr {
            background-color: #f5f5f5;
        }
        .css-18e3th9 {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar design
    logo_url = 'https://ahammadmejbah.com/content/images/2024/10/Mejbah-Ahammad-Profile-8.png'
    logo_encoded = img_to_bytes(logo_url)
    st.sidebar.markdown(
        f"""
        <a href="https://ahammadmejbah.com/">
            <img src='data:image/png;base64,{logo_encoded}' class='img-fluid' width=100>
        </a>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.header('🧰 Scikit-Learn Ultimate Cheat Sheet')
    st.sidebar.markdown('''
    <small>Comprehensive Scikit-Learn commands, functions, and workflows for efficient machine learning and data analysis.</small>
    ''', unsafe_allow_html=True)

    st.sidebar.markdown('__🔑 Key Libraries__')
    st.sidebar.code('''
$ pip install scikit-learn pandas numpy matplotlib seaborn plotly
    ''')

    st.sidebar.markdown('__💡 Tips & Tricks__')
    st.sidebar.code('''
- Always standardize features before training
- Use train_test_split for splitting data
- Cross-validation for model evaluation
- Utilize pipelines for streamlined workflows
- Encode categorical variables appropriately
    ''')

    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Scikit-Learn Cheat Sheet v3.0](https://github.com/ahammadmejbah/Scikit-Learn-Cheat-Sheet) | Nov 2024 | [Mejbah Ahammad](https://ahammadmejbah.com/)</small>''', unsafe_allow_html=True)

    # Load Lottie animations
    lottie_header = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json")
    lottie_footer = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_b8onqq.json")

    # Header with animation
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"""
            <div style="text-align: left; padding: 10px;">
                <h1 style="color: #1f77b4;">🤖 Scikit-Learn Ultimate Cheat Sheet</h1>
                <h3 style="color: #333333;">By Mejbah Ahammad</h3>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        if lottie_header:
            st_lottie(lottie_header, height=150, key="header_animation")
        else:
            st.image("https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png", width=150)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tabs for navigation
    tabs = ["Introduction", "Datasets", "Preprocessing", "Feature Selection", "Regression", "Classification", "Clustering", "Model Evaluation", "Model Selection", "Dimensionality Reduction", "Pipelines", "Model Persistence", "Miscellaneous"]
    tab_icons = ["📖", "📚", "🧹", "🔍", "📈", "🔎", "🌀", "📊", "⚙️", "📉", "🔗", "💾", "🔧"]
    tab_items = [f"{icon} {name}" for icon, name in zip(tab_icons, tabs)]
    selected_tab = st.tabs(tab_items)

    with selected_tab[0]:
        st.markdown("## Introduction to Scikit-Learn")
        st.markdown("""
        **Scikit-Learn** is a powerful Python library for machine learning. It provides a range of supervised and unsupervised learning algorithms via a consistent interface.
        """)
        st.markdown("""
        <img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" alt="Scikit-Learn Logo" width="200">
        """, unsafe_allow_html=True)
        st.markdown("### Key Features")
        st.markdown("""
        - Simple and efficient tools for predictive data analysis
        - Accessible to everybody, and reusable in various contexts
        - Built on NumPy, SciPy, and matplotlib
        - Open source, commercially usable - BSD license
        """)
        st.markdown("### Basic Workflow")
        st.markdown("""
        1. Import the required modules and classes
        2. Load your data and split into training and testing sets
        3. Preprocess your data (scaling, normalization, etc.)
        4. Choose a model and train it
        5. Evaluate the model
        6. Fine-tune your model (hyperparameter tuning)
        """)

    # (Continue with the full content for all tabs as in the previous code.)

    # Footer with animation
    st.markdown("<hr>", unsafe_allow_html=True)
    col_footer1, col_footer2 = st.columns([2,1])
    with col_footer1:
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius:10px;">
                <p>Connect with me:</p>
                <div style="display: flex; gap: 20px;">
                    <a href="https://facebook.com/ahammadmejbah" target="_blank">
                        <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" alt="Facebook" width="30">
                    </a>
                    <a href="https://instagram.com/ahammadmejbah" target="_blank">
                        <img src="https://cdn-icons-png.flaticon.com/512/733/733558.png" alt="Instagram" width="30">
                    </a>
                    <a href="https://github.com/ahammadmejbah" target="_blank">
                        <img src="https://cdn-icons-png.flaticon.com/512/733/733553.png" alt="GitHub" width="30">
                    </a>
                    <a href="https://ahammadmejbah.com/" target="_blank">
                        <img src="https://cdn-icons-png.flaticon.com/512/919/919827.png" alt="Portfolio" width="30">
                    </a>
                </div>
                <br>
                <small>Scikit-Learn Cheat Sheet v3.0 | Nov 2024 | <a href="https://ahammadmejbah.com/" style="color: #1f77b4;">Mejbah Ahammad</a></small>
                <div class="card-footer">Mejbah Ahammad © 2024</div>
            </div>
        """, unsafe_allow_html=True)
    with col_footer2:
        if lottie_footer:
            st_lottie(lottie_footer, height=150, key="footer_animation")
        else:
            st.image("https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png", width=150)

if __name__ == '__main__':
    main()
