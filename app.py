import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# Define a variable for the data directory
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# Define columns as per the document
COLUMNS = ['YEAR', 'MONTH', 'DAY', 'ORDER', 'COUNTRY', 'SESSION_ID',
           'PAGE_1_MAIN_CATEGORY', 'PAGE_2_CLOTHING_MODEL', 'COLOUR',
           'LOCATION', 'MODEL_PHOTOGRAPHY', 'PRICE',
           'PRICE_2_HIGHER_THAN_AVG', 'PAGE']

# --- Data Generation and Loading (Cached) ---
@st.cache_data
def create_and_load_data(num_records=10000):
    """
    Creates a synthetic dataset and saves it to a CSV file.
    """
    st.info("Generating synthetic data for demonstration...")
    np.random.seed(42)

    def create_synthetic_data(num_records, is_test=False):
        data = {
            'YEAR': 2008,
            'MONTH': np.random.choice(range(4, 9), num_records),
            'DAY': np.random.randint(1, 31, num_records),
            'ORDER': np.random.randint(1, 50, num_records),
            'COUNTRY': np.random.randint(1, 48, num_records),
            'SESSION_ID': np.arange(1000, 1000 + num_records),
            'PAGE_1_MAIN_CATEGORY': np.random.randint(1, 5, num_records),
            'PAGE_2_CLOTHING_MODEL': np.random.randint(1, 218, num_records),
            'COLOUR': np.random.randint(1, 15, num_records),
            'LOCATION': np.random.randint(1, 7, num_records),
            'MODEL_PHOTOGRAPHY': np.random.randint(1, 3, num_records),
            'PRICE': np.random.uniform(5, 500, num_records).round(2),
            'PRICE_2_HIGHER_THAN_AVG': np.random.choice([1, 2], num_records),
            'PAGE': np.random.randint(1, 6, num_records),
        }
        df = pd.DataFrame(data)
        
        for col in ['COUNTRY', 'PRICE']:
            df.loc[df.sample(frac=0.02).index, col] = np.nan
        
        df['PURCHASE_COMPLETED'] = np.where(
            (df['PRICE'] < 100) & (df['ORDER'] > 20) & (df['PAGE'] < 3), 1, 2
        )
        df['REVENUE'] = (df['PRICE'] * np.random.uniform(1.1, 1.5, num_records)).round(2)
        
        if is_test:
            df = df.drop(columns=['PURCHASE_COMPLETED', 'REVENUE'])
        
        return df

    # Create and save files
    train_df = create_synthetic_data(num_records=10000)
    test_df = create_synthetic_data(num_records=2000, is_test=True)
    train_df.to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, 'test.csv'), index=False)
    
    return train_df, test_df

# --- Model Training (Cached) ---
@st.cache_resource
def train_models(train_df):
    """
    Trains the classification, regression, and clustering models.
    """
    st.info("Training machine learning models...")
    
    # --- FIX: Robust Data Preprocessing ---
    # Impute missing values for all relevant columns before splitting the data
    train_df['PRICE'].fillna(train_df['PRICE'].median(), inplace=True)
    train_df['COUNTRY'].fillna(train_df['COUNTRY'].mode()[0], inplace=True)

    # Feature Engineering
    train_df['DATE'] = pd.to_datetime(train_df[['YEAR', 'MONTH', 'DAY']])
    train_df['DAY_OF_WEEK'] = train_df['DATE'].dt.dayofweek

    X = train_df.drop(columns=['PURCHASE_COMPLETED', 'REVENUE', 'YEAR', 'DATE'])
    y_classification = train_df['PURCHASE_COMPLETED']
    y_regression = train_df['REVENUE']
    
    # Impute NaNs in the target variables
    y_classification.fillna(y_classification.mode()[0], inplace=True)
    y_regression.fillna(y_regression.median(), inplace=True)

    numerical_features = ['ORDER', 'PRICE', 'PAGE', 'DAY_OF_WEEK']
    categorical_features = ['COUNTRY', 'PAGE_1_MAIN_CATEGORY', 'COLOUR', 'LOCATION', 'MODEL_PHOTOGRAPHY', 'PRICE_2_HIGHER_THAN_AVG']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Classification Pipeline
    cls_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    cls_pipeline.fit(X, y_classification)

    # Regression Pipeline
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    reg_pipeline.fit(X, y_regression)
    
    # Clustering Model
    clustering_features = ['ORDER', 'PRICE', 'PAGE_1_MAIN_CATEGORY', 'COLOUR']
    X_clustering = train_df[clustering_features]
    
    clustering_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['ORDER', 'PRICE']),
            ('cat', OneHotEncoder(), ['PAGE_1_MAIN_CATEGORY', 'COLOUR'])
        ],
        remainder='passthrough'
    )
    X_clustering_processed = clustering_preprocessor.fit_transform(X_clustering)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_clustering_processed)
    train_df['Cluster'] = cluster_labels
    
    return cls_pipeline, reg_pipeline, kmeans, train_df, clustering_preprocessor

# --- Streamlit UI ---
st.set_page_config(page_title="Clickstream Analysis App", layout="wide")

st.title("Customer Conversion & Segmentation App")
st.markdown("### Powered by Machine Learning on Clickstream Data")
st.markdown("This interactive application leverages clickstream data to predict customer behavior and segment users.")

# Load and train models (cached for efficiency)
train_data, test_data = create_and_load_data()
cls_model, reg_model, kmeans_model, train_data_with_clusters, clustering_preprocessor = train_models(train_data.copy())

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("Predict for a New Customer")
    st.markdown("Enter the browsing data for a hypothetical customer to see their predictions.")
    
    with st.form("new_customer_form"):
        # User inputs for a new data point
        order = st.slider("Order Number (in session)", 1, 50, 10)
        country = st.number_input("Country (1-47)", 1, 47, 1)
        main_category = st.selectbox("Main Category", range(1, 5))
        page = st.slider("Page Number", 1, 5, 1)
        price = st.slider("Product Price ($)", 5.0, 500.0, 50.0)
        colour = st.selectbox("Colour (1-14)", range(1, 15))
        location = st.selectbox("Photo Location (1-6)", range(1, 7))
        model_photography = st.selectbox("Model Photography", [1, 2], format_func=lambda x: "En Face" if x == 1 else "Profile")
        price_higher_avg = st.selectbox("Price Higher than Avg", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
        
        submitted = st.form_submit_button("Get Predictions")

# --- Main Content ---
if submitted:
    st.subheader("Prediction Results for New Customer")
    
    # Create a DataFrame from user inputs, including placeholder columns for consistency
    new_data = pd.DataFrame([{
        'ORDER': order,
        'COUNTRY': country,
        'PAGE_1_MAIN_CATEGORY': main_category,
        'COLOUR': colour,
        'LOCATION': location,
        'MODEL_PHOTOGRAPHY': model_photography,
        'PRICE': price,
        'PRICE_2_HIGHER_THAN_AVG': price_higher_avg,
        'PAGE': page,
        'DAY_OF_WEEK': 2, # Placeholder value for a weekday
        'PAGE_2_CLOTHING_MODEL': 1, # Placeholder value
        'SESSION_ID': 999999, # Dummy ID
        'DAY': 1, # Placeholder value
        'MONTH': 1 # Placeholder value
    }])
    
    # Classification Prediction
    cls_prediction = cls_model.predict(new_data)[0]
    purchase_status = "Likely to Purchase" if cls_prediction == 1 else "Unlikely to Purchase"
    
    st.success(f"**Classification:** The customer is **{purchase_status}**.")
    
    # Regression Prediction
    reg_prediction = reg_model.predict(new_data)[0]
    st.info(f"**Revenue Estimation:** The estimated revenue for this customer is **${reg_prediction:.2f}**.")
    
    # Clustering Prediction
    # Create a new DataFrame with only the features used for clustering
    new_data_for_clustering = new_data[['ORDER', 'PRICE', 'PAGE_1_MAIN_CATEGORY', 'COLOUR']]
    new_data_processed = clustering_preprocessor.transform(new_data_for_clustering)
    cluster_prediction = kmeans_model.predict(new_data_processed)[0]
    st.warning(f"**Customer Segment:** This customer belongs to **Cluster {cluster_prediction + 1}**.")

# --- Insights from the Dataset ---
st.header("Dataset Insights & Customer Segmentation")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(train_data):,}")
col2.metric("Purchase Rate", f"{train_data['PURCHASE_COMPLETED'].value_counts(normalize=True)[1]:.2%}")
col3.metric("Average Price", f"${train_data['PRICE'].mean():.2f}")

st.subheader("Customer Segments")
st.markdown("This plot shows how customers are segmented based on their browsing behavior. You can use these insights to tailor marketing campaigns.")

# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=train_data_with_clusters,
    x='ORDER',
    y='PRICE',
    hue='Cluster',
    palette='viridis',
    style='Cluster',
    s=100,
    ax=ax
)
ax.set_title("Customer Segments by Order and Price")
st.pyplot(fig)

st.subheader("Correlation Analysis")
st.markdown("A heatmap to show the relationship between different features.")
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# Show the raw data
if st.checkbox('Show Raw Data'):
    st.subheader("Raw Training Data")
    st.dataframe(train_data.head())
    st.subheader("Raw Test Data")
    st.dataframe(test_data.head())
