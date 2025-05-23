import os
import sys
import pickle

import pandas as pd
import streamlit as st

from data_processing import process_data
from model_training import train_model
from config.paths import get_paths

st.title("Customer Segmentation Prediction")

paths = get_paths()

#
# --- Section 1: Upload & Train ---
#
st.header("1. Upload CSV & Train Model")
uploaded = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded:
    # save raw
    os.makedirs(os.path.dirname(paths["raw_upload"]), exist_ok=True)
    with open(paths["raw_upload"], "wb") as f:
        f.write(uploaded.getbuffer())

    # process → train
    proc = process_data(paths["raw_upload"], paths["processed_data"])
    m = train_model(proc, paths["model"])
    st.success(f"Model trained and saved to: `{m}`")

#
# --- Section 2: Load & Predict ---
#
st.header("2. Load Trained Model & Predict")
if st.button("Load Model"):
    if os.path.exists(paths["model"]):
        with open(paths["model"], "rb") as f:
            st.session_state.model = pickle.load(f)
        st.success("✅ Model loaded into session.")
    else:
        st.error("No model found. Please train first.")

if "model" in st.session_state:
    with st.form("predict"):
        st.subheader("Enter new customer data:")
        # build inputs exactly in the order your processing expects:
        inputs = {
            "Age": st.number_input("Age", min_value=0, value=30),
            "Education": st.number_input("Education (0–4)", min_value=0, max_value=4, value=2),
            "Marital Status": st.selectbox("Married?", (0, 1), help="0=No, 1=Yes"),
            "Parental Status": st.selectbox("Has Kids?", (0, 1)),
            "Children": st.number_input("Number of Children", min_value=0, value=0),
            "Income": st.number_input("Income", min_value=0, value=50_000),
            "Total_Spending": st.number_input("Total Spending", min_value=0, value=2_000),
            "Days_as_Customer": st.number_input("Days as Customer", min_value=0, value=365),
            "Recency": st.number_input("Recency (days since last purchase)", min_value=0, value=30),
            "Wines": st.number_input("Wines Spending", min_value=0, value=500),
            "Fruits": st.number_input("Fruits Spending", min_value=0, value=100),
            "Meat": st.number_input("Meat Spending", min_value=0, value=300),
            "Fish": st.number_input("Fish Spending", min_value=0, value=150),
            "Sweets": st.number_input("Sweets Spending", min_value=0, value=50),
            "Gold": st.number_input("Gold Spending", min_value=0, value=200),
            "Web": st.number_input("Web Purchases", min_value=0, value=5),
            "Catalog": st.number_input("Catalog Purchases", min_value=0, value=3),
            "Store": st.number_input("Store Purchases", min_value=0, value=8),
            "Discount Purchases": st.number_input("Discount Purchases", min_value=0, value=2),
            "Total Promo": st.number_input("Total Promo", min_value=0, value=1),
            "NumWebVisitsMonth": st.number_input("Web Visits/Month", min_value=0, value=4),
        }
        submitted = st.form_submit_button("Predict Cluster")
        if submitted:
            df_in = pd.DataFrame([inputs])
            try:
                cluster = st.session_state.model.predict(df_in)[0]
                st.success(f"🎯 Predicted cluster: **{cluster}**")
            except Exception as e:
                st.error(f"Prediction error: {e}")
