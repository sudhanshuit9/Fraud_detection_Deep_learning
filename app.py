# Streamlit interface for model interaction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import streamlit as st

# Title for the Streamlit interface
st.title("Credit Card Fraud Detection System")

# Sidebar for uploading a CSV dataset
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview")
    st.write(data.head())

    # Preprocessing the data
    X = data.drop('Class', axis=1)  # Features
    y = data['Class']  # Target

    # Scaling 'Amount' and 'Time' features
    scaler = StandardScaler()
    X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Sidebar options for training the model
    if st.sidebar.button('Train Model'):
        # Train RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the model for future use
        joblib.dump(model, 'fraud_detection_model.pkl')
        st.write("Model trained and saved successfully!")

        # Predictions and evaluation
        y_pred = model.predict(X_test)
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        st.write("Classification Report:")
        st.write(classification_report(y_test, y_pred))

        st.write("Accuracy Score:")
        st.write(accuracy_score(y_test, y_pred))

    # Load pre-trained model
    if st.sidebar.button('Load Model'):
        model = joblib.load('fraud_detection_model.pkl')
        st.write("Model loaded successfully!")

        # Predict on the test data
        y_pred = model.predict(X_test)
        st.write("Predictions on test data:")
        st.write(y_pred)

    # Real-time fraud detection on new data
    st.sidebar.title("Fraud Detection")
    uploaded_test_file = st.sidebar.file_uploader("Upload transaction file for detection", type="csv")

    if uploaded_test_file is not None:
        test_data = pd.read_csv(uploaded_test_file)
        st.write("Transaction Data Preview")
        st.write(test_data.head())

        # Scale the 'Amount' and 'Time' features
        test_data['Amount'] = scaler.transform(test_data['Amount'].values.reshape(-1, 1))
        test_data['Time'] = scaler.transform(test_data['Time'].values.reshape(-1, 1))

        # Load the model and make predictions
        model = joblib.load('fraud_detection_model.pkl')
        predictions = model.predict(test_data)

        # Show predictions
        st.write("Fraud Detection Results:")
        st.write(predictions)

