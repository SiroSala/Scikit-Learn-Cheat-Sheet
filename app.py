import streamlit as st
import base64
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page configuration
st.set_page_config(
    page_title='🤖 Scikit-Learn Cheat Sheet by Mejbah Ahammad',
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
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        .main {
            background-color: #f5f5f5;
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
    st.sidebar.header('🧰 Scikit-Learn Cheat Sheet')
    st.sidebar.markdown('''
    <small>Essential Scikit-Learn commands, functions, and workflows for efficient machine learning and data analysis.</small>
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
    ''')

    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Scikit-Learn Cheat Sheet v1.0](https://github.com/ahammadmejbah/Scikit-Learn-Cheat-Sheet) | Nov 2024 | [Mejbah Ahammad](https://ahammadmejbah.com/)</small>''', unsafe_allow_html=True)

    # Load Lottie animations
    lottie_header = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json")
    lottie_footer = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_b8onqq.json")

    # Header with animation
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"""
            <div style="text-align: left; padding: 10px;">
                <h1 style="color: #1f77b4;">🤖 Scikit-Learn Cheat Sheet</h1>
                <h3 style="color: #333333;">By Mejbah Ahammad</h3>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st_lottie(lottie_header, height=150, key="header_animation")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Tabs for navigation
    tabs = ["Introduction", "Importing Data", "Preprocessing", "Regression", "Classification", "Clustering", "Model Evaluation", "Model Selection", "Dimensionality Reduction", "Advanced Topics"]
    tab_icons = ["📖", "📥", "🧹", "📈", "🔍", "🌀", "📊", "🔗", "📉", "🚀"]
    tab_items = [f"{icon} {name}" for icon, name in zip(tab_icons, tabs)]
    selected_tab = st.selectbox("Navigate", tab_items)

    if selected_tab == tab_items[0]:
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

    elif selected_tab == tab_items[1]:
        st.markdown("## 📥 Importing Data")
        st.markdown("### Loading Datasets")
        st.code("""
from sklearn import datasets

# Load built-in datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
        """, language='python')
        st.markdown("### Creating Datasets")
        st.code("""
from sklearn.datasets import make_regression, make_classification

# Regression dataset
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=0.1)

# Classification dataset
X_clf, y_clf = make_classification(n_samples=100, n_features=4, n_classes=2)
        """, language='python')
        st.markdown("### Loading from CSV")
        st.code("""
import pandas as pd

df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']
        """, language='python')
        st.markdown("### Splitting Data")
        st.code("""
from sklearn.model_selection import train_test_split

# For features X and target y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """, language='python')

    elif selected_tab == tab_items[2]:
        st.markdown("## 🧹 Preprocessing")
        st.markdown("### Scaling Features")
        st.code("""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
        """, language='python')
        st.markdown("### Encoding Categorical Variables")
        st.code("""
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
        """, language='python')
        st.markdown("### Imputing Missing Values")
        st.code("""
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
        """, language='python')
        st.markdown("### Pipeline for Preprocessing")
        st.code("""
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

X_processed = pipeline.fit_transform(X)
        """, language='python')

    elif selected_tab == tab_items[3]:
        st.markdown("## 📈 Regression")
        st.markdown("### Linear Regression")
        st.code("""
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
        """, language='python')
        st.markdown("### Evaluating Regression Models")
        st.code("""
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
        """, language='python')
        st.markdown("### Regularized Regression")
        st.code("""
from sklearn.linear_model import Ridge, Lasso

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
        """, language='python')
        st.markdown("### Polynomial Regression")
        st.code("""
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
        """, language='python')

    elif selected_tab == tab_items[4]:
        st.markdown("## 🔍 Classification")
        st.markdown("### Logistic Regression")
        st.code("""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
        """, language='python')
        st.markdown("### Evaluating Classification Models")
        st.code("""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
        """, language='python')
        st.markdown("### Decision Trees")
        st.code("""
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
        """, language='python')
        st.markdown("### Random Forest")
        st.code("""
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
        """, language='python')
        st.markdown("### Support Vector Machines")
        st.code("""
from sklearn.svm import SVC

model = SVC(kernel='rbf')
model.fit(X_train, y_train)
        """, language='python')

    elif selected_tab == tab_items[5]:
        st.markdown("## 🌀 Clustering")
        st.markdown("### K-Means Clustering")
        st.code("""
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_
        """, language='python')
        st.markdown("### Hierarchical Clustering")
        st.code("""
from sklearn.cluster import AgglomerativeClustering

agglo = AgglomerativeClustering(n_clusters=3)
labels = agglo.fit_predict(X)
        """, language='python')
        st.markdown("### DBSCAN Clustering")
        st.code("""
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
        """, language='python')
        st.markdown("### Evaluating Clustering")
        st.code("""
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
        """, language='python')

    elif selected_tab == tab_items[6]:
        st.markdown("## 📊 Model Evaluation")
        st.markdown("### Cross-Validation")
        st.code("""
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
        """, language='python')
        st.markdown("### Grid Search")
        st.code("""
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

best_params = grid.best_params_
best_score = grid.best_score_
        """, language='python')
        st.markdown("### Learning Curves")
        st.code("""
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
        """, language='python')
        st.markdown("### ROC Curve")
        st.code("""
from sklearn.metrics import roc_curve, auc

probs = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
roc_auc = auc(fpr, tpr)
        """, language='python')

    elif selected_tab == tab_items[7]:
        st.markdown("## 🔗 Model Selection")
        st.markdown("### Pipeline")
        st.code("""
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(StandardScaler(), SVC())
pipeline.fit(X_train, y_train)
        """, language='python')
        st.markdown("### Feature Selection")
        st.code("""
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X, y)
        """, language='python')
        st.markdown("### Recursive Feature Elimination")
        st.code("""
from sklearn.feature_selection import RFE

rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
        """, language='python')
        st.markdown("### Ensemble Methods")
        st.code("""
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
model3 = SVC(probability=True)

ensemble = VotingClassifier(estimators=[
    ('lr', model1), ('dt', model2), ('svc', model3)
], voting='soft')

ensemble.fit(X_train, y_train)
        """, language='python')

    elif selected_tab == tab_items[8]:
        st.markdown("## 📉 Dimensionality Reduction")
        st.markdown("### Principal Component Analysis (PCA)")
        st.code("""
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
        """, language='python')
        st.markdown("### t-SNE")
        st.code("""
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
        """, language='python')
        st.markdown("### LDA")
        st.code("""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
        """, language='python')
        st.markdown("### Isomap")
        st.code("""
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2)
X_iso = isomap.fit_transform(X)
        """, language='python')

    elif selected_tab == tab_items[9]:
        st.markdown("## 🚀 Advanced Topics")
        st.markdown("### Hyperparameter Tuning")
        st.code("""
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
random_search = RandomizedSearchCV(SVC(), param_distributions, cv=5)
random_search.fit(X_train, y_train)
        """, language='python')
        st.markdown("### Handling Imbalanced Data")
        st.code("""
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
        """, language='python')
        st.markdown("### Custom Transformers")
        st.code("""
from sklearn.base import TransformerMixin, BaseEstimator

class CustomTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, param=1):
        self.param = param

    def fit(self, X, y=None):
        # Fit logic
        return self

    def transform(self, X):
        # Transform logic
        return X_transformed

# Use in pipeline
pipeline = Pipeline([
    ('custom', CustomTransformer(param=2)),
    ('model', LogisticRegression())
])
        """, language='python')
        st.markdown("### Saving and Loading Models")
        st.code("""
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
model = joblib.load('model.pkl')
        """, language='python')
        st.markdown("### Working with Large Datasets")
        st.code("""
from sklearn.utils import shuffle

# Partial fitting
from sklearn.linear_model import SGDClassifier

model = SGDClassifier()
for X_batch, y_batch in get_data_in_batches():
    model.partial_fit(X_batch, y_batch, classes=np.unique(y))
        """, language='python')

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
                <small>Scikit-Learn Cheat Sheet v1.0 | Nov 2024 | <a href="https://ahammadmejbah.com/" style="color: #1f77b4;">Mejbah Ahammad</a></small>
                <div class="card-footer">Mejbah Ahammad © 2024</div>
            </div>
        """, unsafe_allow_html=True)
    with col_footer2:
        st_lottie(lottie_footer, height=150, key="footer_animation")

if __name__ == '__main__':
    main()
