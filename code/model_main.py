# code/model_main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
from config.paths import get_paths
import sys
# Add this to the beginning of datamain() and modelmain()
paths = get_paths()
os.makedirs(os.path.dirname(paths['raw_upload']), exist_ok=True)
os.makedirs(os.path.dirname(paths['processed_data']), exist_ok=True)
os.makedirs(os.path.dirname(paths['model_path']), exist_ok=True)
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def modelmain(processed_data_path):
    """Train a model on the processed data and return the model file path."""
    paths = get_paths()
    
    # Load processed data
    df = pd.read_csv(processed_data_path)
    
    # Split features and target (cluster)
    X = df.drop(columns=['cluster'])
    y = df['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Hyperparameter tuning for logistic regression
    params = {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [100, 200, 300],
    }
    
    model_cv = RandomizedSearchCV(
        LogisticRegression(),
        param_distributions=params,
        cv=5,
        n_iter=10,
        n_jobs=-1,
        random_state=42
    )
    model_cv.fit(X_train, y_train)
    
    # Save the best model
    os.makedirs(os.path.dirname(paths['model_path']), exist_ok=True)
    with open(paths['model_path'], "wb") as f:
        pickle.dump(model_cv.best_estimator_, f)
    
    # Print training results
    print(f"Best parameters: {model_cv.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, model_cv.predict(X_test)):.2f}")
    
    return paths['model_path']
