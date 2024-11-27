import streamlit as st
import base64
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title='📊 Pandas Cheat Sheet by Mejbah Ahammad',
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
    ds_sidebar()
    ds_body()

def ds_sidebar():
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
    st.sidebar.header('🧰 Pandas Cheat Sheet')
    
    st.sidebar.markdown('''
    <small>Essential Pandas commands, functions, and workflows for efficient data manipulation and analysis.</small>
    ''', unsafe_allow_html=True)
    
    st.sidebar.markdown('__🔑 Key Libraries__')
    st.sidebar.code('''
$ pip install pandas numpy matplotlib seaborn plotly
    ''')
    
    st.sidebar.markdown('__💡 Tips & Tricks__')
    st.sidebar.code('''
- Always check data types with `df.dtypes`
- Use vectorized operations for efficiency
- Handle missing data with `df.isnull()` and `df.fillna()`
- Utilize `groupby` for aggregation
    ''')
    
    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[Pandas Cheat Sheet v1.0](https://github.com/ahammadmejbah/Pandas-Cheat-Sheet) | Nov 2024 | [Mejbah Ahammad](https://ahammadmejbah.com/)<div class="card-footer">Mejbah Ahammad © 2024</div></small>''', unsafe_allow_html=True)

def ds_body():
    # Load Lottie animations
    lottie_header = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json")
    lottie_intro = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_tfb3estd.json")
    lottie_footer = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_b8onqq.json")
    
    # Header with animation
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown(f"""
            <div style="text-align: left; padding: 10px;">
                <h1 style="color: #1f77b4;">📊 Pandas Cheat Sheet</h1>
                <h3 style="color: #333333;">By Mejbah Ahammad</h3>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st_lottie(lottie_header, height=150, key="header_animation")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Introduction Section with animation
    st.markdown("## Introduction to Pandas")
    col_intro1, col_intro2 = st.columns([2,1])
    with col_intro1:
        st.markdown("""
        **Pandas** is a powerful Python library for data manipulation and analysis. It provides data structures like **DataFrame** and **Series** to work with structured data seamlessly.
        """)
    with col_intro2:
        st_lottie(lottie_intro, height=150, key="intro_animation")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Pandas Basics
    st.markdown("## 📁 Pandas Basics")
    st.markdown("### Importing Pandas")
    st.code("""
import pandas as pd
    """, language='python')
    
    st.markdown("### Creating DataFrames")
    st.code("""
# From a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

# From a CSV file
df = pd.read_csv('data.csv')
    """, language='python')
    
    st.markdown("### Viewing Data")
    st.code("""
# View first few rows
df.head()

# View last few rows
df.tail()

# DataFrame info
df.info()

# Summary statistics
df.describe()
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Data Selection
    st.markdown("## 🔍 Data Selection")
    st.markdown("### Selecting Columns")
    st.code("""
# Single column
df['Age']

# Multiple columns
df[['Name', 'Age']]
    """, language='python')
    
    st.markdown("### Selecting Rows")
    st.code("""
# By index
df.iloc[0:5]

# By condition
df[df['Age'] > 30]

# Using loc
df.loc[df['City'] == 'New York']

# Using iloc
df.iloc[[0, 2, 4]]
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Data Cleaning
    st.markdown("## 🧹 Data Cleaning")
    st.markdown("### Handling Missing Values")
    st.code("""
# Drop missing values
df.dropna(inplace=True)

# Fill missing values
df.fillna(value=0, inplace=True)
    """, language='python')
    
    st.markdown("### Removing Duplicates")
    st.code("""
df.drop_duplicates(inplace=True)
    """, language='python')
    
    st.markdown("### Data Type Conversion")
    st.code("""
df['Age'] = df['Age'].astype(int)
    """, language='python')
    
    st.markdown("### Renaming Columns")
    st.code("""
df.rename(columns={'Name': 'Full Name'}, inplace=True)
    """, language='python')
    
    st.markdown("### Replacing Values")
    st.code("""
df['City'].replace({'New York': 'NY', 'Los Angeles': 'LA'}, inplace=True)
    """, language='python')
    
    st.markdown("### Filtering Outliers")
    st.code("""
df = df[df['Salary'] < df['Salary'].quantile(0.95)]
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Data Transformation
    st.markdown("## 🔄 Data Transformation")
    st.markdown("### Applying Functions")
    st.code("""
# Apply a function to a column
df['Age'] = df['Age'].apply(lambda x: x + 1)
    """, language='python')
    
    st.markdown("### Vectorized Operations")
    st.code("""
df['Salary'] = df['Salary'] * 1.1
    """, language='python')
    
    st.markdown("### Mapping Values")
    st.code("""
mapping = {'New York': 'NY', 'Los Angeles': 'LA', 'Chicago': 'CHI'}
df['City'] = df['City'].map(mapping)
    """, language='python')
    
    st.markdown("### Binning")
    st.code("""
df['Age Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    """, language='python')
    
    st.markdown("### Creating New Columns")
    st.code("""
df['Salary_Per_Age'] = df['Salary'] / df['Age']
    """, language='python')
    
    st.markdown("### String Operations")
    st.code("""
df['Name'] = df['Name'].str.upper()
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Merging & Joining
    st.markdown("## 🔗 Merging & Joining")
    st.markdown("### Merging DataFrames")
    st.code("""
merged_df = pd.merge(df1, df2, on='Key', how='inner')
    """, language='python')
    
    st.markdown("### Concatenating DataFrames")
    st.code("""
concatenated_df = pd.concat([df1, df2], axis=0)
    """, language='python')
    
    st.markdown("### Joining DataFrames")
    st.code("""
joined_df = df1.join(df2, how='inner')
    """, language='python')
    
    st.markdown("### Merging on Multiple Keys")
    st.code("""
merged_df = pd.merge(df1, df2, on=['Key1', 'Key2'], how='outer')
    """, language='python')
    
    st.markdown("### Merge with Indicator")
    st.code("""
merged_df = pd.merge(df1, df2, on='Key', how='outer', indicator=True)
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Grouping & Aggregation
    st.markdown("## 📊 Grouping & Aggregation")
    st.markdown("### Group By")
    st.code("""
grouped = df.groupby('City')

# Aggregation
grouped['Age'].mean()
    """, language='python')
    
    st.markdown("### Multiple Aggregations")
    st.code("""
grouped.agg({'Age': ['mean', 'sum'], 'Salary': 'median'})
    """, language='python')
    
    st.markdown("### Group By with Multiple Columns")
    st.code("""
grouped = df.groupby(['City', 'Age Group'])
    """, language='python')
    
    st.markdown("### Aggregation with Custom Functions")
    st.code("""
grouped.agg({
    'Salary': ['mean', 'sum'],
    'Experience': lambda x: x.max() - x.min()
})
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Pivot Tables
    st.markdown("## 📈 Pivot Tables")
    st.markdown("### Creating a Pivot Table")
    st.code("""
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', fill_value=0)
    """, language='python')
    
    st.markdown("### Multiple Aggregation Functions in Pivot Tables")
    st.code("""
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc=['sum', 'mean'], fill_value=0)
    """, language='python')
    
    st.markdown("### Adding Margins to Pivot Tables")
    st.code("""
pivot = df.pivot_table(values='Sales', index='Region', columns='Product', aggfunc='sum', margins=True, fill_value=0)
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Data Visualization
    st.markdown("## 📈 Data Visualization")
    st.markdown("### Matplotlib")
    st.code("""
import matplotlib.pyplot as plt

# Line Plot
plt.figure(figsize=(10,5))
plt.plot(x, y, label='Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot')
plt.legend()
plt.grid(True)
plt.show()

# Bar Chart
plt.figure(figsize=(10,5))
plt.bar(categories, values, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart')
plt.show()

# Scatter Plot
plt.figure(figsize=(10,5))
plt.scatter(x, y, color='red')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()

# Histogram
plt.figure(figsize=(10,5))
plt.hist(data, bins=10, color='green', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Pie Chart
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart')
plt.axis('equal')
plt.show()
    """, language='python')
    
    st.markdown("### Seaborn")
    st.code("""
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter Plot with Regression Line
sns.lmplot(x='Age', y='Salary', data=df, aspect=1.5)
plt.title('Age vs Salary with Regression Line')
plt.show()

# Heatmap
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='City', y='Salary', data=df)
plt.title('Salary Distribution by City')
plt.show()

# Pairplot
sns.pairplot(df, hue='City')
plt.show()

# Violin Plot
plt.figure(figsize=(10,6))
sns.violinplot(x='City', y='Salary', data=df)
plt.title('Salary Distribution by City')
plt.show()
    """, language='python')
    
    st.markdown("### Plotly")
    st.code("""
import plotly.express as px

# Scatter Plot
fig = px.scatter(df, x='Age', y='Salary', color='City', title='Age vs Salary by City')
fig.show()

# Bar Chart
fig = px.bar(df, x='City', y='Sales', color='City', barmode='group', title='Sales by City')
fig.show()

# Line Chart
fig = px.line(df, x='Date', y='Sales', title='Sales Over Time')
fig.show()

# Histogram
fig = px.histogram(df, x='Age', nbins=10, title='Age Distribution')
fig.show()

# Pie Chart
fig = px.pie(df, names='Product', values='Sales', title='Sales Distribution by Product')
fig.show()
    """, language='python')
    
    st.markdown("### Plotly Express Example")
    st.code("""
# Interactive Scatter Plot
fig = px.scatter(df, x='Age', y='Salary', color='City', hover_data=['Name'], title='Interactive Age vs Salary')
st.plotly_chart(fig)

# Interactive Bar Chart
fig = px.bar(df, x='City', y='Sales', color='City', barmode='group', title='Interactive Sales by City')
st.plotly_chart(fig)

# Interactive Line Chart
fig = px.line(df, x='Date', y='Sales', title='Interactive Sales Over Time')
st.plotly_chart(fig)

# Interactive Histogram
fig = px.histogram(df, x='Age', nbins=10, title='Interactive Age Distribution')
st.plotly_chart(fig)

# Interactive Pie Chart
fig = px.pie(df, names='Product', values='Sales', title='Interactive Sales Distribution by Product')
st.plotly_chart(fig)
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Advanced Pandas
    st.markdown("## 🚀 Advanced Pandas")
    
    st.markdown("### Time Series Handling")
    st.code("""
# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set index
df.set_index('Date', inplace=True)

# Resample data
monthly = df.resample('M').mean()

# Rolling window
df['Rolling_Mean'] = df['Sales'].rolling(window=3).mean()
    """, language='python')
    
    st.markdown("### Performance Optimization")
    st.code("""
# Use categorical data types
df['Category'] = df['Category'].astype('category')

# Use vectorized operations instead of apply
df['Salary'] = df['Salary'] * 1.1

# Avoid loops by using apply or map
df['New_Column'] = df['Existing_Column'].map(lambda x: x * 2)
    """, language='python')
    
    st.markdown("### Working with Large Datasets")
    st.code("""
# Use chunksize to read large files
chunks = pd.read_csv('large_data.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)

# Use Dask for parallel processing
import dask.dataframe as dd
ddf = dd.read_csv('large_data.csv')
result = ddf.groupby('Category').mean().compute()
    """, language='python')
    
    st.markdown("### MultiIndex")
    st.code("""
# Creating MultiIndex
arrays = [
    ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'],
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame({'A': range(8), 'B': range(8)}, index=index)

# Accessing data
df.loc['bar', 'one']
df.xs('one', level='second')
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Data Visualization with Pandas
    st.markdown("## 📊 Data Visualization with Pandas")
    
    st.markdown("### Line Plot")
    st.code("""
# Line Plot
df['Sales'].plot(kind='line', figsize=(10,5), title='Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
    """, language='python')
    
    st.markdown("### Bar Chart")
    st.code("""
# Bar Chart
df.groupby('Category')['Sales'].sum().plot(kind='bar', color='skyblue', figsize=(10,5), title='Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()
    """, language='python')
    
    st.markdown("### Scatter Plot")
    st.code("""
# Scatter Plot
df.plot(kind='scatter', x='Age', y='Salary', color='red', figsize=(10,5), title='Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
    """, language='python')
    
    st.markdown("### Histogram")
    st.code("""
# Histogram
df['Age'].plot(kind='hist', bins=10, color='green', edgecolor='black', figsize=(10,5), title='Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
    """, language='python')
    
    st.markdown("### Pie Chart")
    st.code("""
# Pie Chart
df.groupby('Category')['Sales'].sum().plot(kind='pie', autopct='%1.1f%%', figsize=(8,8), title='Sales Distribution by Category')
plt.ylabel('')
plt.show()
    """, language='python')
    
    st.markdown("### Seaborn Integration")
    st.code("""
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

# Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(x='Category', y='Sales', data=df)
plt.title('Sales Distribution by Category')
plt.show()
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Interactive Visualization
    st.markdown("## 🎨 Interactive Visualization")
    st.markdown("### Plotly Express in Streamlit")
    st.code("""
# Interactive Scatter Plot
fig = px.scatter(df, x='Age', y='Salary', color='City', title='Interactive Age vs Salary')
st.plotly_chart(fig)

# Interactive Bar Chart
fig = px.bar(df, x='Category', y='Sales', color='Category', barmode='group', title='Interactive Sales by Category')
st.plotly_chart(fig)

# Interactive Line Chart
fig = px.line(df, x='Date', y='Sales', title='Interactive Sales Over Time')
st.plotly_chart(fig)

# Interactive Histogram
fig = px.histogram(df, x='Age', nbins=10, title='Interactive Age Distribution')
st.plotly_chart(fig)

# Interactive Pie Chart
fig = px.pie(df, names='Product', values='Sales', title='Interactive Sales Distribution by Product')
st.plotly_chart(fig)
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Advanced Topics
    st.markdown("## 🧠 Advanced Topics")
    
    st.markdown("### Time Series Analysis")
    st.code("""
# Resampling
monthly = df['Sales'].resample('M').sum()

# Rolling Window
df['Rolling_Mean'] = df['Sales'].rolling(window=3).mean()

# Time Series Plot
monthly.plot(kind='line', figsize=(10,5), title='Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()
    """, language='python')
    
    st.markdown("### Performance Optimization")
    st.code("""
# Use categorical data types
df['Category'] = df['Category'].astype('category')

# Vectorized operations instead of apply
df['Salary'] = df['Salary'] * 1.1

# Avoid loops by using apply or map
df['New_Column'] = df['Existing_Column'].map(lambda x: x * 2)
    """, language='python')
    
    st.markdown("### Working with Large Datasets")
    st.code("""
# Read large CSV in chunks
chunks = pd.read_csv('large_data.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)

# Using Dask for parallel processing
import dask.dataframe as dd
ddf = dd.read_csv('large_data.csv')
result = ddf.groupby('Category').mean().compute()
    """, language='python')
    
    st.markdown("### MultiIndex")
    st.code("""
# Creating MultiIndex
arrays = [
    ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
    ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'],
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame({'A': range(8), 'B': range(8)}, index=index)

# Accessing data
df.loc['bar', 'one']
df.xs('one', level='second')
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Advanced Visualization
    st.markdown("## 🎨 Advanced Visualization")
    
    st.markdown("### Pairplot with Seaborn")
    st.code("""
# Pairplot
sns.pairplot(df, hue='Category')
plt.show()
    """, language='python')
    
    st.markdown("### Jointplot with Seaborn")
    st.code("""
# Jointplot
sns.jointplot(x='Age', y='Salary', data=df, kind='scatter')
plt.show()
    """, language='python')
    
    st.markdown("### FacetGrid with Seaborn")
    st.code("""
# FacetGrid
g = sns.FacetGrid(df, col="Category", hue="City")
g.map(plt.scatter, "Age", "Salary").add_legend()
plt.show()
    """, language='python')
    
    st.markdown("### PairGrid with Seaborn")
    st.code("""
# PairGrid
g = sns.PairGrid(df, hue="Category")
g.map_diag(plt.hist)
g.map_offdiag(sns.scatterplot)
g.add_legend()
plt.show()
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Interactive Widgets
    st.markdown("## 🛠 Interactive Widgets")
    
    st.markdown("### DataFrame Display with Filters")
    st.code("""
# Interactive DataFrame display
import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv('data.csv')

# Sidebar filters
category = st.sidebar.multiselect('Select Category', df['Category'].unique())
city = st.sidebar.multiselect('Select City', df['City'].unique())

# Apply filters
filtered_df = df[
    (df['Category'].isin(category)) &
    (df['City'].isin(city))
]

# Display DataFrame
st.dataframe(filtered_df)
    """, language='python')
    
    st.markdown("### Dynamic Plotting")
    st.code("""
# Dynamic plotting based on user selection
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# User selects plot type
plot_type = st.selectbox('Select Plot Type', ['Line', 'Bar', 'Scatter', 'Histogram'])

# User selects columns
x_col = st.selectbox('Select X-axis', df.columns)
y_col = st.selectbox('Select Y-axis', df.columns)

# Plot based on selection
if plot_type == 'Line':
    df.plot(kind='line', x=x_col, y=y_col, figsize=(10,5))
    plt.title(f'Line Plot of {y_col} over {x_col}')
    plt.show()
elif plot_type == 'Bar':
    df.plot(kind='bar', x=x_col, y=y_col, figsize=(10,5))
    plt.title(f'Bar Chart of {y_col} by {x_col}')
    plt.show()
elif plot_type == 'Scatter':
    df.plot(kind='scatter', x=x_col, y=y_col, figsize=(10,5))
    plt.title(f'Scatter Plot of {y_col} vs {x_col}')
    plt.show()
elif plot_type == 'Histogram':
    df[y_col].plot(kind='hist', bins=10, figsize=(10,5))
    plt.title(f'Histogram of {y_col}')
    plt.xlabel(y_col)
    plt.show()
    """, language='python')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Footer with Social Media Links and Animation
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
                <small>Data Science Cheat Sheet v1.0 | Nov 2024 | <a href="https://ahammadmejbah.com/" style="color: #1f77b4;">Mejbah Ahammad</a></small>
                <div class="card-footer">Mejbah Ahammad © 2024</div>
            </div>
        """, unsafe_allow_html=True)
    with col_footer2:
        st_lottie(lottie_footer, height=150, key="footer_animation")
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
