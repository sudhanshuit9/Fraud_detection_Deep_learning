import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# App Title
st.title("Dynamic Fraud Detection System")

# Upload Section
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Display the uploaded data
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.write(data.head())
    
    # Fraud data visualization
    if 'Class' in data.columns:
        st.subheader("Fraud vs. Non-Fraud Distribution")
        fraud_count = data['Class'].value_counts()
        labels = ['Non-Fraud', 'Fraud']
        fig, ax = plt.subplots()
        ax.pie(fraud_count, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'red'])
        ax.axis('equal')
        st.pyplot(fig)

# Input Transaction Details for Single Prediction
st.sidebar.header("Input Transaction Details")
input_form = st.sidebar.form("transaction_form")
features = {}

if uploaded_file:
    for col in data.columns[:-1]:  # Exclude the target column
        features[col] = input_form.number_input(f"Enter {col}", value=0.0)

    submitted = input_form.form_submit_button("Submit Transaction")
    if submitted:
        st.write("Transaction Details Submitted for Prediction:")
        st.write(features)

# Predict Button
if uploaded_file and st.sidebar.button("Predict Fraud for Dataset"):
    # Call the backend API for prediction
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": uploaded_file}
    )
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        st.subheader("Predictions for Uploaded Dataset")
        data["Predicted Fraud"] = predictions
        st.write(data.head())

        # Display fraud predictions
        st.subheader("Predicted Fraud Distribution")
        pred_fraud_count = pd.Series(predictions).value_counts()
        labels = ['Non-Fraud', 'Fraud']
        fig, ax = plt.subplots()
        ax.pie(pred_fraud_count, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'orange'])
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.error("Error in API call. Check the backend server.")

# App Footer
st.sidebar.info("Developed by Sudhanshu Singh")
