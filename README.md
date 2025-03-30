# Customer Segmentation with Streamlit

This project implements customer segmentation using machine learning. It includes a Streamlit web app that allows users to predict a customer's cluster based on their attributes, such as age, spending habits, and marital status.

## Project Structure

```
├── data_processing.py           # Data preprocessing script
├── model_main.py                # Model training and saving script
├── app.py                       # Streamlit app for customer segmentation
├── models/
│   └── best_logistic_regression_model.pkl  # Saved machine learning model
├── README.md                    # Project documentation
```

## Features
- Preprocesses customer data for machine learning.
- Trains a logistic regression model for clustering customers.
- Saves the trained model for reuse.
- Provides a Streamlit app interface for predicting customer clusters.

## Requirements

Ensure you have the following installed:
- Python 3.8+
- Streamlit
- Scikit-learn
- NumPy

Install dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to Use

### Step 1: Train the Model
Run the `model_main.py` script to preprocess the data, train the model, and save it as a `.pkl` file:

```bash
python model_main.py
```

The trained model will be saved in the `models/` directory as `best_logistic_regression_model.pkl`.

### Step 2: Run the Streamlit App
Start the Streamlit app to predict customer clusters:

```bash
streamlit run app.py
```

### Step 3: Predict Clusters
1. Open the app in your browser (default: `http://localhost:8501`).
2. Enter the customer details in the provided input fields.
3. Click the **Predict Cluster** button to get the predicted cluster for the customer.

## File Descriptions

### `data_processing.py`
Handles data preprocessing tasks such as cleaning, scaling, and encoding features.

### `model_main.py`
1. Loads and preprocesses the data using `data_processing.py`.
2. Trains a logistic regression model.
3. Saves the trained model to `models/best_logistic_regression_model.pkl`.

### `app.py`
1. Loads the pre-trained model from the `models/` directory.
2. Provides a web interface to input customer data and predict clusters.
3. Displays the predicted cluster based on the provided input.

## Example Input for the App
- **Age:** 30
- **Education Level:** Level 2
- **Marital Status:** Married
- **Parental Status:** Yes
- **Number of Children:** 1
- **Income:** 50000
- **Total Spending:** 2000
- **Days as Customer:** 365
- **Recency (Days):** 30
- **Wines Spending:** 500
- **Fruits Spending:** 100
- **Meat Spending:** 300
- **Fish Spending:** 150
- **Sweets Spending:** 50
- **Gold Spending:** 200
- **Web Purchases:** 5
- **Catalog Purchases:** 3
- **Store Purchases:** 8
- **Discount Purchases:** 2
- **Total Promo:** 1
- **Number of Web Visits per Month:** 4

## Troubleshooting

- **Model file not found:**
  Ensure that `best_logistic_regression_model.pkl` is present in the `models/` directory. If not, run `model_main.py` to train and save the model.

- **App crashes or doesn't start:**
  Ensure all required libraries are installed. Use the command:
  ```bash
  pip install -r requirements.txt
  ```




