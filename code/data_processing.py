# code/data_processing.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
from config.paths import get_paths
import sys
# Add this to the beginning of datamain() and modelmain()
paths = get_paths()
os.makedirs(os.path.dirname(paths['raw_upload']), exist_ok=True)
os.makedirs(os.path.dirname(paths['processed_data']), exist_ok=True)
os.makedirs(os.path.dirname(paths['model_path']), exist_ok=True)
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)

def load_data(filepath, sep=','):
    """Load the dataset from a file."""
    return pd.read_csv(filepath, sep=sep)

def preprocess_data(df):
    """Clean and transform the raw data."""
    df = df.copy()
    
    # Fill missing Income values
    df['Income'].fillna(df['Income'].median(), inplace=True)
    
    # Create new feature for Age using Year_Birth
    current_year = datetime.now().year
    df['Age'] = current_year - df['Year_Birth']
    
    # Convert Education to numeric levels
    df["Education"].replace({
        "Basic": 0, 
        "2n Cycle": 1, 
        "Graduation": 2, 
        "Master": 3, 
        "PhD": 4
    }, inplace=True)
    
    # Convert Marital_Status to binary (1 for married/together, 0 otherwise)
    df['Marital_Status'].replace({
        "Married": 1, "Together": 1, "Absurd": 0, 
        "Widow": 0, "YOLO": 0, "Divorced": 0, 
        "Single": 0, "Alone": 0
    }, inplace=True)
    
    # Create children and family size features
    df['Children'] = df['Kidhome'] + df['Teenhome']
    df['Family_Size'] = df['Marital_Status'] + df['Children'] + 1
    
    # Create total spending and total promo features
    df['Total_Spending'] = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + 
                            df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"])
    df["Total Promo"] = (df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] + 
                         df["AcceptedCmp4"] + df["AcceptedCmp5"])
    
    # Calculate days as customer
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    today = datetime.today()
    df['Days_as_Customer'] = (today - df['Dt_Customer']).dt.days
    
    # Create parental status indicator
    df["Parental Status"] = np.where(df["Children"] > 0, 1, 0)
    
    # Drop columns that are no longer needed
    columns_to_drop = ['Year_Birth', 'Kidhome', 'Teenhome']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    # Rename columns to align with model features
    df.rename(columns={
        "Marital_Status": "Marital Status",
        "MntWines": "Wines",
        "MntFruits": "Fruits",
        "MntMeatProducts": "Meat",
        "MntFishProducts": "Fish",
        "MntSweetProducts": "Sweets",
        "MntGoldProds": "Gold",
        "NumWebPurchases": "Web",
        "NumCatalogPurchases": "Catalog",
        "NumStorePurchases": "Store",
        "NumDealsPurchases": "Discount Purchases",
        "NumWebVisitsMonth": "NumWebVisitsMonth"
    }, inplace=True)
    
    return df[["Age", "Education", "Marital Status", "Parental Status", "Children", "Income", 
               "Total_Spending", "Days_as_Customer", "Recency", "Wines", "Fruits", "Meat", 
               "Fish", "Sweets", "Gold", "Web", "Catalog", "Store", "Discount Purchases", 
               "Total Promo", "NumWebVisitsMonth"]]

def detect_outliers(df, continuous_features):
    """Detect and handle outliers in the dataset."""
    for col in continuous_features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        df.loc[df[col] > upper_limit, col] = upper_limit
        df.loc[df[col] < lower_limit, col] = lower_limit
    return df

def preprocess_pipeline(df):
    """Create preprocessing pipelines for numeric and outlier features."""
    num_features = [col for col in df.columns if df[col].dtype != 'O']
    continuous_features = [col for col in num_features if df[col].nunique() > 25]
    df = detect_outliers(df, continuous_features)
    
    outlier_features = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Age", "Total_Spending"]
    numeric_features = [col for col in num_features if col not in outlier_features]
    
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value=0)),
        ("scaler", StandardScaler())
    ])
    
    outlier_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='constant', fill_value=0)),
        ("transformer", PowerTransformer(standardize=True))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("outlier", outlier_pipeline, outlier_features)
    ])
    return preprocessor

def apply_pca(scaled_data, n_components=2):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    return pd.DataFrame(pca.fit_transform(scaled_data),
                        columns=[f'PC{i+1}' for i in range(n_components)])

def cluster_data(pcadf, n_clusters=3):
    """Perform KMeans clustering and return cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(pcadf)

def save_data(df, output_path):
    """Save the processed DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def datamain(uploaded_file):
    """Process the uploaded CSV file and return the path to the clean, clustered data."""
    paths = get_paths()
    
    # Save the uploaded file
    os.makedirs(os.path.dirname(paths['raw_upload']), exist_ok=True)
    with open(paths['raw_upload'], "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and preprocess the data
    df = load_data(paths['raw_upload'])
    df = preprocess_data(df)
    
    # Apply preprocessing pipeline
    preprocessor = preprocess_pipeline(df)
    scaled_data = preprocessor.fit_transform(df)
    
    # Apply PCA and clustering
    pcadf = apply_pca(scaled_data)
    df["cluster"] = cluster_data(pcadf)
    
    # Save the processed data
    save_data(df, paths['processed_data'])
    return paths['processed_data']
