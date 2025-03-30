from data_processing import datamain
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import pickle
import os

def modelmain():
    # Execute the data preprocessing pipeline
    datamain()

    # Load the clustered data
    df = pd.read_csv('../data/clustered_data.csv')

    # Split the data into training and test sets
    X = df.drop(columns=['cluster'])
    y = df['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Hyperparameter tuning
    params = {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [100, 200, 300],
    }
    model = RandomizedSearchCV(LogisticRegression(), param_distributions=params, cv=5, n_iter=10)
    model.fit(X_train, y_train)

    # Evaluate the model
    print("Best Parameters:", model.best_params_)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save the model
    os.makedirs("../models", exist_ok=True)
    with open("../models/best_logistic_regression_model.pkl", "wb") as f:
        pickle.dump(model.best_estimator_, f)
    print("Model saved!")

if __name__ == "__main__":
    modelmain()
