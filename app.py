import streamlit as st
import base64
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                     GridSearchCV, RandomizedSearchCV, 
                                     StratifiedKFold, learning_curve, 
                                     TimeSeriesSplit)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                   OneHotEncoder, LabelEncoder, PolynomialFeatures, 
                                   Normalizer, Binarizer, KBinsDiscretizer, 
                                   OrdinalEncoder)
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
                              load_boston, load_wine, load_diabetes, 
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

    with selected_tab[1]:
        st.markdown("## 📚 Datasets")
        st.markdown("### Loading Built-in Datasets")
        st.code("""
from sklearn.datasets import load_iris, load_digits, load_boston, load_wine, load_breast_cancer

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Load Digits dataset
digits = load_digits()
X = digits.data
y = digits.target
        """, language='python')
        st.markdown("### Creating Synthetic Datasets")
        st.code("""
from sklearn.datasets import make_classification, make_regression, make_blobs

# Classification dataset
X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=2)

# Regression dataset
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1)

# Clustering dataset
X_cluster, y_cluster = make_blobs(n_samples=1000, centers=3, n_features=2)
        """, language='python')
        st.markdown("### Loading External Datasets")
        st.code("""
import pandas as pd

# Load dataset from CSV
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']
        """, language='python')
        st.markdown("### Splitting Data")
        st.code("""
from sklearn.model_selection import train_test_split

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        """, language='python')

    with selected_tab[2]:
        st.markdown("## 🧹 Preprocessing")
        st.markdown("### Scaling Features")
        st.code("""
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-Max Scaling
minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

# Robust Scaling
robust = RobustScaler()
X_train_robust = robust.fit_transform(X_train)
X_test_robust = robust.transform(X_test)
        """, language='python')
        st.markdown("### Encoding Categorical Variables")
        st.code("""
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = ohe.fit_transform(X_categorical)

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y_categorical)

# Ordinal Encoding
oe = OrdinalEncoder()
X_ordinal = oe.fit_transform(X_categorical)
        """, language='python')
        st.markdown("### Imputing Missing Values")
        st.code("""
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# Simple Imputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)
X_knn_imputed = knn_imputer.fit_transform(X)

# Iterative Imputer
iter_imputer = IterativeImputer(random_state=0)
X_iter_imputed = iter_imputer.fit_transform(X)
        """, language='python')
        st.markdown("### Generating Polynomial Features")
        st.code("""
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)
        """, language='python')
        st.markdown("### Normalization and Binarization")
        st.code("""
from sklearn.preprocessing import Normalizer, Binarizer

# Normalization
normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X)

# Binarization
binarizer = Binarizer(threshold=0.0)
X_binarized = binarizer.fit_transform(X)
        """, language='python')
        st.markdown("### Discretization")
        st.code("""
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile')
X_binned = discretizer.fit_transform(X)
        """, language='python')

    with selected_tab[3]:
        st.markdown("## 🔍 Feature Selection")
        st.markdown("### Univariate Feature Selection")
        st.code("""
from sklearn.feature_selection import SelectKBest, chi2, f_classif

# For classification
selector = SelectKBest(score_func=chi2, k=10)
X_new = selector.fit_transform(X, y)

# For regression
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)
        """, language='python')
        st.markdown("### Recursive Feature Elimination")
        st.code("""
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
        """, language='python')
        st.markdown("### Recursive Feature Elimination with Cross-Validation")
        st.code("""
from sklearn.feature_selection import RFECV

model = LogisticRegression()
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
X_rfecv = rfecv.fit_transform(X, y)
        """, language='python')
        st.markdown("### SelectFromModel")
        st.code("""
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
selector = SelectFromModel(model, prefit=True)
X_selected = selector.transform(X)
        """, language='python')
        st.markdown("### Variance Threshold")
        st.code("""
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
X_var = selector.fit_transform(X)
        """, language='python')

    with selected_tab[4]:
        st.markdown("## 📈 Regression")
        st.markdown("### Linear Regression")
        st.code("""
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
        """, language='python')
        st.markdown("### Regularized Regression")
        st.code("""
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# ElasticNet Regression
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
        """, language='python')
        st.markdown("### Polynomial Regression")
        st.code("""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
model = LinearRegression()
model.fit(X_poly, y_train)
y_pred = model.predict(poly.transform(X_test))
        """, language='python')
        st.markdown("### Support Vector Regression")
        st.code("""
from sklearn.svm import SVR

svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
        """, language='python')
        st.markdown("### Decision Tree Regression")
        st.code("""
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
        """, language='python')
        st.markdown("### Ensemble Regression")
        st.code("""
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Random Forest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

# Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=100)
gbr.fit(X_train, y_train)

# AdaBoost
ada = AdaBoostRegressor(n_estimators=100)
ada.fit(X_train, y_train)
        """, language='python')
        st.markdown("### Neural Network Regression")
        st.code("""
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
        """, language='python')

    with selected_tab[5]:
        st.markdown("## 🔎 Classification")
        st.markdown("### Logistic Regression")
        st.code("""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
        """, language='python')
        st.markdown("### K-Nearest Neighbors")
        st.code("""
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
        """, language='python')
        st.markdown("### Support Vector Machines")
        st.code("""
from sklearn.svm import SVC

svc = SVC(kernel='rbf', probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
        """, language='python')
        st.markdown("### Decision Trees")
        st.code("""
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
        """, language='python')
        st.markdown("### Random Forest")
        st.code("""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
        """, language='python')
        st.markdown("### Naive Bayes")
        st.code("""
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
        """, language='python')
        st.markdown("### Gradient Boosting")
        st.code("""
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
        """, language='python')
        st.markdown("### Neural Network Classification")
        st.code("""
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
        """, language='python')
        st.markdown("### Evaluation Metrics")
        st.code("""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
        """, language='python')

    with selected_tab[6]:
        st.markdown("## 🌀 Clustering")
        st.markdown("### K-Means Clustering")
        st.code("""
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
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
        st.markdown("### Mean Shift Clustering")
        st.code("""
from sklearn.cluster import MeanShift

ms = MeanShift()
labels = ms.fit_predict(X)
        """, language='python')
        st.markdown("### Spectral Clustering")
        st.code("""
from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters=3)
labels = sc.fit_predict(X)
        """, language='python')
        st.markdown("### OPTICS Clustering")
        st.code("""
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=5)
labels = optics.fit_predict(X)
        """, language='python')
        st.markdown("### BIRCH Clustering")
        st.code("""
from sklearn.cluster import Birch

birch = Birch(n_clusters=3)
labels = birch.fit_predict(X)
        """, language='python')
        st.markdown("### Evaluating Clustering")
        st.code("""
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

silhouette = silhouette_score(X, labels)
calinski = calinski_harabasz_score(X, labels)
davies = davies_bouldin_score(X, labels)
        """, language='python')

    with selected_tab[7]:
        st.markdown("## 📊 Model Evaluation")
        st.markdown("### Regression Metrics")
        st.code("""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
        """, language='python')
        st.markdown("### Classification Metrics")
        st.code("""
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        """, language='python')
        st.markdown("### Cross-Validation")
        st.code("""
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
mean_score = scores.mean()
        """, language='python')
        st.markdown("### Learning Curves")
        st.code("""
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
        """, language='python')
        st.markdown("### ROC and AUC")
        st.code("""
from sklearn.metrics import roc_curve, auc

probs = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probs[:,1])
roc_auc = auc(fpr, tpr)
        """, language='python')
        st.markdown("### Precision-Recall Curve")
        st.code("""
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, probs[:,1])
        """, language='python')

    with selected_tab[8]:
        st.markdown("## ⚙️ Model Selection")
        st.markdown("### Grid Search")
        st.code("""
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
best_params = grid.best_params_
best_score = grid.best_score_
        """, language='python')
        st.markdown("### Randomized Search")
        st.code("""
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, None]}
random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions, n_iter=5, cv=5)
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
        """, language='python')
        st.markdown("### Cross-Validation with StratifiedKFold")
        st.code("""
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
        """, language='python')
        st.markdown("### Ensemble Methods")
        st.code("""
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression()
model2 = RandomForestClassifier()
model3 = SVC(probability=True)

ensemble = VotingClassifier(estimators=[
    ('lr', model1), ('rf', model2), ('svc', model3)
], voting='soft')

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
        """, language='python')
        st.markdown("### Stacking")
        st.code("""
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('svc', SVC(kernel='linear', probability=True))
]

clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
        """, language='python')

    with selected_tab[9]:
        st.markdown("## 📉 Dimensionality Reduction")
        st.markdown("### Principal Component Analysis (PCA)")
        st.code("""
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
        """, language='python')
        st.markdown("### t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        st.code("""
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
        """, language='python')
        st.markdown("### Linear Discriminant Analysis (LDA)")
        st.code("""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
        """, language='python')
        st.markdown("### Truncated SVD")
        st.code("""
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)
        """, language='python')
        st.markdown("### Non-negative Matrix Factorization (NMF)")
        st.code("""
from sklearn.decomposition import NMF

nmf = NMF(n_components=2)
X_nmf = nmf.fit_transform(X)
        """, language='python')
        st.markdown("### Isomap")
        st.code("""
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2)
X_iso = isomap.fit_transform(X)
        """, language='python')
        st.markdown("### Locally Linear Embedding")
        st.code("""
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2)
X_lle = lle.fit_transform(X)
        """, language='python')
        st.markdown("### Spectral Embedding")
        st.code("""
from sklearn.manifold import SpectralEmbedding

se = SpectralEmbedding(n_components=2)
X_se = se.fit_transform(X)
        """, language='python')

    with selected_tab[10]:
        st.markdown("## 🔗 Pipelines")
        st.markdown("### Creating a Pipeline")
        st.code("""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
        """, language='python')
        st.markdown("### FeatureUnion")
        st.code("""
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD

# Combine features
combined_features = FeatureUnion([
    ('pca', PCA(n_components=2)),
    ('svd', TruncatedSVD(n_components=2))
])

X_features = combined_features.fit_transform(X)
        """, language='python')
        st.markdown("### ColumnTransformer")
        st.code("""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ['age', 'income']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['gender', 'city']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

clf.fit(X_train, y_train)
        """, language='python')

    with selected_tab[11]:
        st.markdown("## 💾 Model Persistence")
        st.markdown("### Saving and Loading Models with Joblib")
        st.code("""
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
model = joblib.load('model.pkl')
        """, language='python')
        st.markdown("### Saving and Loading Models with Pickle")
        st.code("""
import pickle

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
        """, language='python')

    with selected_tab[12]:
        st.markdown("## 🔧 Miscellaneous")
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
        st.markdown("### Handling Text Data")
        st.code("""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Count Vectorizer
cv = CountVectorizer()
X_cv = cv.fit_transform(text_data)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(text_data)
        """, language='python')
        st.markdown("### Time Series Forecasting")
        st.code("""
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
        """, language='python')
        st.markdown("### Anomaly Detection")
        st.code("""
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1)
iso_forest.fit(X)
anomalies = iso_forest.predict(X)
        """, language='python')
        st.markdown("### Multi-label Classification")
        st.code("""
from sklearn.multiclass import OneVsRestClassifier

ovr = OneVsRestClassifier(SVC())
ovr.fit(X_train, y_train)
y_pred = ovr.predict(X_test)
        """, language='python')
        st.markdown("### Model Interpretation with SHAP")
        st.code("""
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
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
