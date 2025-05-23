import os
import pickle

import pandas as pd
import streamlit as st

from data_processing import load_and_preprocess
from model_training import train_kmeans, MODEL_FILE

st.title("🛍️ Customer Clustering Demo")

# 1) UPLOAD & CLUSTER
uploaded = st.file_uploader("Upload your marketing TSV (tab-separated) file", type="csv")
if uploaded:
    # save raw
    raw = 'uploaded.csv'
    with open(raw, 'wb') as f:
        f.write(uploaded.getbuffer())

    # preprocess & cluster
    X = load_and_preprocess(raw)
    km = train_kmeans(X)
    labels = km.labels_
    X['cluster'] = labels

    st.success("✅ Clustering complete!")
    st.write("First 10 rows with assigned cluster:")
    st.write(X.head(10))

# 2) PREDICT A NEW POINT (OPTIONAL)
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        km = pickle.load(f)

    st.header("Predict cluster for a new customer")
    cols = ['Age', 'Income', 'Recency', 'NumWebVisitsMonth']
    vals = [st.number_input(c, min_value=0, value=0) for c in cols]

    if st.button("Predict"):
        new_df = pd.DataFrame([vals], columns=cols)
        cl = km.predict(new_df)[0]
        st.success(f"Predicted cluster: **{cl}**")
else:
    st.info("▶️ Upload & cluster first to create the model.")
