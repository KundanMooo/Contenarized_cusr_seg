# code/app.py
# In app.py
from code.data_processing import datamain
from code.model_main import modelmain
from config.paths import get_paths

import sys
import os

# Add the 'code' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))


import streamlit as st
import pickle
import pandas as pd
import numpy as np
from data_processing import datamain
from model_main import modelmain
from config.paths import get_paths
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Debug prints to verify environment paths
st.write("### Current Python Path")
st.write(sys.path)
st.write("### Current Directory")
st.write(os.getcwd())
st.write("### Config Path Exists?")
st.write(os.path.exists(os.path.join(sys.path[0], 'config')))

# Initialize session state for model and timestamp
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
    st.session_state.model_timestamp = None

def main():
    st.title("Customer Segmentation Prediction")
    
    # File upload and model training section
    with st.expander("Upload New Data & Train Model", expanded=True):
        uploaded_file = st.file_uploader("Upload raw CSV data", type=["csv"])
        if uploaded_file:
            with st.spinner("Processing data and training new model..."):
                try:
                    # Process raw CSV and get processed data path
                    processed_path = datamain(uploaded_file)
                    # Train the model on processed data
                    model_path = modelmain(processed_path)
                    
                    # Load the newly trained model
                    with open(model_path, "rb") as f:
                        st.session_state.current_model = pickle.load(f)
                    st.session_state.model_timestamp = os.path.basename(model_path).split("_")[1]
                    st.success(f"New model trained successfully! (Timestamp: {st.session_state.model_timestamp})")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
    
    # Prediction form for customer input
    st.header("Make Prediction")
    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        education = st.selectbox("Education Level", options=[0, 1, 2, 3, 4], 
                                 format_func=lambda x: ["Basic", "2n Cycle", "Graduation", "Master", "PhD"][x])
        marital_status = st.selectbox("Marital Status", options=[0, 1], 
                                      format_func=lambda x: "Married" if x == 1 else "Single")
        parental_status = st.selectbox("Parental Status", options=[0, 1], 
                                       format_func=lambda x: "Yes" if x == 1 else "No")
        children = st.number_input("Number of Children", min_value=0, value=0)
        income = st.number_input("Income", min_value=0, value=50000)
        total_spending = st.number_input("Total Spending", min_value=0, value=2000)
        days_as_customer = st.number_input("Days as Customer", min_value=0, value=365)
        recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)
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
            if st.session_state.current_model:
                try:
                    # Create a DataFrame with the expected feature names
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
                    
                    # Make prediction using the trained model
                    prediction = st.session_state.current_model.predict(input_data)
                    st.success(f"Predicted Cluster: **{prediction[0]}**")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
            else:
                st.warning("Please upload data and train a model first!")

if __name__ == "__main__":
    main()
