import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_model(processed_data_path: str, model_path: str) -> str:
    """
    Loads the processed CSV, trains a LogisticRegression on the 'cluster' label,
    and pickles the model to `model_path`.
    """
    df = pd.read_csv(processed_data_path)
    X = df.drop(columns=["cluster"])
    y = df["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path
