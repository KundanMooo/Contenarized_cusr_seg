# app.py
import streamlit as st
import os
import sys
import pickle
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.abspath('.'))

from data_processing import process_data
from model_main import train_model
from config.paths import get_paths

st.title("Customer Segmentation Prediction")

# --- Section 1: Upload CSV and Train Model ---
st.header("Upload CSV & Train Model")
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    paths = get_paths()
    raw_path = paths["raw_upload"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process data using our new process_data function and get the processed file path
    processed_path = process_data(raw_path)
    
    # Train the model using the processed data
    model_path = train_model(processed_path)
    st.success(f"Model trained successfully and saved at: {model_path}")

# --- Section 2: Make Prediction ---
st.header("Make a Prediction")
if st.button("Load Trained Model"):
    # Load the model into session state (if available)
    model_file = os.path.join("models", "model.pkl")  # Using a fixed file name from model_training.py
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            st.session_state.current_model = pickle.load(f)
        st.success("Model loaded successfully!")
    else:
        st.error("No trained model found. Please upload and train a model first.")

if "current_model" in st.session_state:
    with st.form("prediction_form"):
        st.subheader("Enter the Input Values")
        # Input fields (adjust as needed to match your processed data columns)
        age = st.number_input("Age", min_value=0, value=30)
        education = st.number_input("Education (numeric code)", min_value=0, value=2)
        marital_status = st.number_input("Marital Status (0=Single, 1=Married)", min_value=0, max_value=1, value=0)
        parental_status = st.number_input("Parental Status (0=No, 1=Yes)", min_value=0, max_value=1, value=0)
        children = st.number_input("Number of Children", min_value=0, value=0)
        income = st.number_input("Income", min_value=0, value=50000)
        total_spending = st.number_input("Total Spending", min_value=0, value=2000)
        days_as_customer = st.number_input("Days as Customer", min_value=0, value=365)
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
        wines = st.number_input("Wines Spending", min_value=0, value=500)
        fruits = st.number_input("Fruits Spending", min_value=0, value=100)
        meat = st.number_input("Meat Spending", min_value=0, value=300)
        fish = st.number_input("Fish Spending", min_value=0, value=150)
        sweets = st.number_input("Sweets Spending", min_value=0, value=50)
        gold = st.number_input("Gold Spending", min_value=0, value=200)
        web = st.number_input("Web Purchases", min_value=0, value=5)
        catalog = st.number_input("Catalog Purchases", min_value=0, value=3)
        store = st.number_input("Store Purchases", min_value=0, value=8)
        discount_purchases = st.number_input("Discount Purchases", min_value=0, value=2)
        total_promo = st.number_input("Total Promo", min_value=0, value=1)
        num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, value=4)
        
        submitted = st.form_submit_button("Predict Cluster")
        if submitted:
            # Build the input dataframe
            input_data = pd.DataFrame([[
                age, education, marital_status, parental_status, children,
                income, total_spending, days_as_customer, recency, wines,
                fruits, meat, fish, sweets, gold, web, catalog, store,
                discount_purchases, total_promo, num_web_visits
            ]], columns=[
                'Age', 'Education', 'Marital Status', 'Parental Status', 'Children',
                'Income', 'Total_Spending', 'Days_as_Customer', 'Recency', 'Wines',
                'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold', 'Web', 'Catalog',
                'Store', 'Discount Purchases', 'Total Promo', 'NumWebVisitsMonth'
            ])
            
            try:
                prediction = st.session_state.current_model.predict(input_data)
                st.success(f"Predicted Cluster: {prediction[0]}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
else:
    st.info("Please load a trained model first by clicking the 'Load Trained Model' button.")
