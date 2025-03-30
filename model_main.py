# model_training.py
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(processed_data_path):
    """
    Train a simple predictive model on the processed data.
    The target is the 'cluster' column.
    """
    # Load the processed data
    df = pd.read_csv(processed_data_path)
    
    # Separate features and target label
    X = df.drop(columns=["cluster"])
    y = df["cluster"]
    
    # Split the data (minimal split here)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a logistic regression classifier (this is just for demonstration)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    # Save the trained model in the models directory (using a fixed file name)
    model_file = os.path.join("models", "model.pkl")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    
    return model_file
