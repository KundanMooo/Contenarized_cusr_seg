import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


def load_data(filepath: str, sep: str = "\t") -> pd.DataFrame:
    return pd.read_csv(filepath, sep=sep)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) fill Income
    df["Income"].fillna(df["Income"].median(), inplace=True)

    # 2) feature engineering
    df["Age"] = 2022 - df["Year_Birth"]
    df["Education"] = df["Education"].map(
        {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
    )
    df["Marital Status"] = df["Marital_Status"].map(
        {
            "Married": 1,
            "Together": 1,
            "Absurd": 0,
            "Widow": 0,
            "YOLO": 0,
            "Divorced": 0,
            "Single": 0,
            "Alone": 0,
        }
    )
    df["Children"] = df["Kidhome"] + df["Teenhome"]
    df["Parental Status"] = (df["Children"] > 0).astype(int)
    df["Family_Size"] = df["Marital Status"] + df["Children"] + 1

    # total spending & promos
    df["Total_Spending"] = df[
        [
            "MntWines",
            "MntFruits",
            "MntMeatProducts",
            "MntFishProducts",
            "MntSweetProducts",
            "MntGoldProds",
        ]
    ].sum(axis=1)
    df["Total Promo"] = df[
        ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    ].sum(axis=1)

    # days as customer
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    df["Days_as_Customer"] = (datetime.today() - df["Dt_Customer"]).dt.days

    # drop everything we won’t use
    df.drop(
        columns=[
            "Year_Birth",
            "Marital_Status",
            "Kidhome",
            "Teenhome",
            "Dt_Customer",
            "Response",
            "Complain",
            "Z_CostContact",
            "Z_Revenue",
            "AcceptedCmp1",
            "AcceptedCmp2",
            "AcceptedCmp3",
            "AcceptedCmp4",
            "AcceptedCmp5",
        ],
        inplace=True,
        errors="ignore",
    )

    # rename to shorter names
    df.rename(
        columns={
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
        },
        inplace=True,
    )

    # final column order
    keep = [
        "Age",
        "Education",
        "Marital Status",
        "Parental Status",
        "Children",
        "Income",
        "Total_Spending",
        "Days_as_Customer",
        "Recency",
        "Wines",
        "Fruits",
        "Meat",
        "Fish",
        "Sweets",
        "Gold",
        "Web",
        "Catalog",
        "Store",
        "Discount Purchases",
        "Total Promo",
        "NumWebVisitsMonth",
    ]
    return df[keep]


def detect_outliers(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in features:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def preprocess_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    # all numeric
    num_feats = df.columns.tolist()
    # choose a handful to PowerTransform, the rest just scale
    outlier_feats = ["Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Age", "Total_Spending"]
    normal_feats = [c for c in num_feats if c not in outlier_feats]

    # clip outliers first
    df2 = detect_outliers(df, outlier_feats)

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)), ("scale", StandardScaler())])
    out_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)), ("pw", PowerTransformer())])

    return ColumnTransformer(
        [
            ("norm", num_pipe, normal_feats),
            ("out", out_pipe, outlier_feats),
        ]
    )


def process_data(input_filepath: str, output_path: str) -> str:
    """
    1) loads raw CSV
    2) preprocesses
    3) scales → PCA(2) → KMeans(3)
    4) writes out processed + cluster
    """
    df = load_data(input_filepath, sep="\t")
    df = preprocess_data(df)

    pre = preprocess_pipeline(df)
    scaled = pre.fit_transform(df)  # → (n_samples, n_features)

    # PCA dims → 2
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(scaled)  # → (n_samples, 2)

    # cluster
    labels = KMeans(n_clusters=3, random_state=42).fit_predict(pcs)

    # write back: original features + cluster
    out = df.copy()
    out["cluster"] = labels

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    return output_path
