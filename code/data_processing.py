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

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 100)

def load_data(filepath, sep='\t'):
    """Load the dataset from a file."""
    return pd.read_csv(filepath, sep=sep)

def preprocess_data(df):
    """Preprocess the data: handle missing values, create new features, and drop unnecessary columns."""
    df = df.copy()
    
    # Handle missing values
    df.Income.fillna(df.Income.median(), inplace=True)
    
    # Create new features
    df['Age'] = 2022 - df['Year_Birth']
    df["Education"].replace({"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}, inplace=True)
    df['Marital_Status'].replace({"Married": 1, "Together": 1, "Absurd": 0, "Widow": 0, "YOLO": 0, "Divorced": 0, 
                                   "Single": 0, "Alone": 0}, inplace=True)
    df['Children'] = df['Kidhome'] + df['Teenhome']
    df['Family_Size'] = df['Marital_Status'] + df['Children'] + 1
    df['Total_Spending'] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + \
                           df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    df["Total Promo"] = df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"] + \
                        df["AcceptedCmp4"] + df["AcceptedCmp5"]
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
    today = datetime.today()
    df['Days_as_Customer'] = (today - df['Dt_Customer']).dt.days
    df['Offers_Responded_To'] = df['AcceptedCmp1'] + df['AcceptedCmp2'] + df['AcceptedCmp3'] + \
                                df['AcceptedCmp4'] + df['AcceptedCmp5'] + df['Response']
    df["Parental Status"] = np.where(df["Children"] > 0, 1, 0)
    
    # Drop unnecessary columns
    columns_to_drop = ['Year_Birth', 'Kidhome', 'Teenhome']
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    # Rename columns
    df.rename(columns={"Marital_Status": "Marital Status", "MntWines": "Wines", "MntFruits": "Fruits",
                       "MntMeatProducts": "Meat", "MntFishProducts": "Fish", "MntSweetProducts": "Sweets",
                       "MntGoldProds": "Gold", "NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog",
                       "NumStorePurchases": "Store", "NumDealsPurchases": "Discount Purchases"},
              inplace=True)
    
    # Filter relevant columns
    df = df[["Age", "Education", "Marital Status", "Parental Status", "Children", "Income", "Total_Spending", 
             "Days_as_Customer", "Recency", "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Web", 
             "Catalog", "Store", "Discount Purchases", "Total Promo", "NumWebVisitsMonth"]]
    
    return df

def detect_outliers(df, continuous_features):
    """Detect and handle outliers in the dataset."""
    for col in continuous_features:
        percentile25 = df[col].quantile(0.25)
        percentile75 = df[col].quantile(0.75)
        iqr = percentile75 - percentile25
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        df.loc[df[col] > upper_limit, col] = upper_limit
        df.loc[df[col] < lower_limit, col] = lower_limit
    return df

def preprocess_pipeline(df):
    """Create preprocessing pipelines for numeric and outlier features."""
    num_features = [feature for feature in df.columns if df[feature].dtype != 'O']
    continuous_features = [feature for feature in num_features if len(df[feature].unique()) > 25]
    df = detect_outliers(df, continuous_features)

    outlier_features = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Age", "Total_Spending"]
    numeric_features = [x for x in num_features if x not in outlier_features]

    numeric_pipeline = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy='constant', fill_value=0)),
        ("StandardScaler", StandardScaler())
    ])

    outlier_features_pipeline = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy='constant', fill_value=0)),
        ("transformer", PowerTransformer(standardize=True))
    ])

    preprocessor = ColumnTransformer([
        ("numeric_pipeline", numeric_pipeline, numeric_features),
        ("outlier_features_pipeline", outlier_features_pipeline, outlier_features)
    ])
    return preprocessor

def apply_pca(scaled_data, n_components=2):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    return pd.DataFrame(pca.fit_transform(scaled_data), columns=[f'PC{i+1}' for i in range(n_components)])

def cluster_data(pcadf, n_clusters=3):
    """Perform KMeans clustering and return cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42).fit(pcadf)
    return model.labels_

def save_data(df, output_path="../data/clustered_data.csv"):
    """Save the processed DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def datamain():
    filepath = '../data/marketing_campaign.csv'
    df = load_data(filepath)
    df = preprocess_data(df)

    preprocessor = preprocess_pipeline(df)
    scaled_data = preprocessor.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

    pcadf = apply_pca(scaled_df)
    df["cluster"] = cluster_data(pcadf)

    save_data(df)

if __name__ == "__main__":
    datamain()
