import streamlit as st
import pickle
import numpy as np

# Function to load the saved model
@st.cache_resource
def load_model(filename='../models/best_logistic_regression_model.pkl'):
    try:
        with open(filename, 'rb') as model_file:
            return pickle.load(model_file)
    except FileNotFoundError:
        st.error(f"Model file not found at {filename}. Ensure the file path is correct.")
        return None

# Function to predict the cluster
def predict_cluster(input_data, model):
    try:
        prediction = model.predict(input_data)
        return prediction[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Streamlit Web App
def main():
    st.title("Customer Segmentation Prediction")
    st.write("Enter the customer details below to predict their cluster:")

    # Input fields for the features
    age = st.number_input("Age", min_value=0, max_value=100, step=1, value=30)
    education = st.selectbox("Education Level", options=[0, 1, 2, 3], format_func=lambda x: f"Level {x}")
    marital_status = st.selectbox("Marital Status", options=[0, 1], format_func=lambda x: "Married" if x == 1 else "Single")
    parental_status = st.selectbox("Parental Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    children = st.number_input("Number of Children", min_value=0, step=1, value=0)
    income = st.number_input("Income", min_value=0, step=1000, value=50000)
    total_spending = st.number_input("Total Spending", min_value=0, step=100, value=2000)
    days_as_customer = st.number_input("Days as Customer", min_value=0, step=1, value=365)
    recency = st.number_input("Recency (Days since last purchase)", min_value=0, step=1, value=30)
    wines = st.number_input("Wines Spending", min_value=0, step=10, value=500)
    fruits = st.number_input("Fruits Spending", min_value=0, step=10, value=100)
    meat = st.number_input("Meat Spending", min_value=0, step=10, value=300)
    fish = st.number_input("Fish Spending", min_value=0, step=10, value=150)
    sweets = st.number_input("Sweets Spending", min_value=0, step=10, value=50)
    gold = st.number_input("Gold Spending", min_value=0, step=10, value=200)
    web = st.number_input("Web Purchases", min_value=0, step=1, value=5)
    catalog = st.number_input("Catalog Purchases", min_value=0, step=1, value=3)
    store = st.number_input("Store Purchases", min_value=0, step=1, value=8)
    discount_purchases = st.number_input("Discount Purchases", min_value=0, step=1, value=2)
    total_promo = st.number_input("Total Promo", min_value=0, step=1, value=1)
    num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, step=1, value=4)

    # Predict button
    if st.button("Predict Cluster"):
        # Combine user inputs into a single array
        input_data = np.array([[age, education, marital_status, parental_status, children,
                                income, total_spending, days_as_customer, recency, wines,
                                fruits, meat, fish, sweets, gold, web, catalog, store,
                                discount_purchases, total_promo, num_web_visits]])

        # Load the model
        model = load_model()

        if model is not None:
            # Predict the cluster
            cluster = predict_cluster(input_data, model)

            if cluster is not None:
                # Display the result
                st.success(f"The predicted cluster for the given input is: **Cluster {cluster}**")
            else:
                st.error("Unable to generate a prediction. Check input values or model integrity.")
        else:
            st.error("Model loading failed. Ensure the model is trained and saved correctly.")

if __name__ == "__main__":
    main()
