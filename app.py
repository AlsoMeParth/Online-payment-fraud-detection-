<<<<<<< HEAD
import streamlit as st
import joblib
import pandas as pd

# Load model, features, and scaler
model, feature_names, X_train_mean, X_train_std = joblib.load('model_bundle.pkl')

# Preprocessing function
def preprocess_input(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, trans_type):
    # Manual one-hot encoding for transaction type
    type_CASH_OUT = 1 if trans_type == 'CASH_OUT' else 0
    type_DEBIT = 1 if trans_type == 'DEBIT' else 0
    type_PAYMENT = 1 if trans_type == 'PAYMENT' else 0
    type_TRANSFER = 1 if trans_type == 'TRANSFER' else 0
    type_CASH_IN = 1 if trans_type == 'CASH_IN' else 0

    # Create DataFrame with input features
    df = pd.DataFrame([{
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type_CASH_OUT': type_CASH_OUT,
        'type_DEBIT': type_DEBIT,
        'type_PAYMENT': type_PAYMENT,
        'type_TRANSFER': type_TRANSFER,
        'type_CASH_IN': type_CASH_IN
    }])

    # Add engineered features
    df['errorOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    df['errorDest'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']
    df['is_anomalous'] = ((df['amount'] > 0) & 
                          (df['oldbalanceOrg'] == 0) & 
                          (df['newbalanceOrig'] == 0) & 
                          (df['oldbalanceDest'] == 0) & 
                          (df['newbalanceDest'] == 0)).astype(int)

    # Reindex to match model's trained feature names
    df = df.reindex(feature_names, axis=1, fill_value=0)

    return df

# Streamlit app
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Online Payment Fraud Detection")
st.write("Use the sidebar to input transaction details and check for fraud detection results.")

# Sidebar inputs
st.sidebar.header("Transaction Details")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, format="%.2f")
trans_type = st.sidebar.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]).strip().upper()
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, format="%.2f")
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, format="%.2f")
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, format="%.2f")
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, format="%.2f")

# Predict button
if st.sidebar.button("Check for Fraud"):
    # Preprocess input
    input_df = preprocess_input(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, trans_type)

    # Scale input
    input_scaled = (input_df - X_train_mean) / X_train_std

    # Convert to NumPy array to avoid feature name warning
    input_scaled_array = input_scaled.to_numpy()

    # Predict
    prediction = model.predict(input_scaled_array)[0]

    # Display result
    result = "Fraud" if prediction == 1 else "Not Fraud"
    if result == "Fraud":
        st.error(f"Prediction: {result}")
    else:
=======
import streamlit as st
import joblib
import pandas as pd

# Load model, features, and scaler
model, feature_names, X_train_mean, X_train_std = joblib.load('model_bundle.pkl')

# Preprocessing function
def preprocess_input(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, trans_type):
    # Manual one-hot encoding for transaction type
    type_CASH_OUT = 1 if trans_type == 'CASH_OUT' else 0
    type_DEBIT = 1 if trans_type == 'DEBIT' else 0
    type_PAYMENT = 1 if trans_type == 'PAYMENT' else 0
    type_TRANSFER = 1 if trans_type == 'TRANSFER' else 0
    type_CASH_IN = 1 if trans_type == 'CASH_IN' else 0

    # Create DataFrame with input features
    df = pd.DataFrame([{
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'type_CASH_OUT': type_CASH_OUT,
        'type_DEBIT': type_DEBIT,
        'type_PAYMENT': type_PAYMENT,
        'type_TRANSFER': type_TRANSFER,
        'type_CASH_IN': type_CASH_IN
    }])

    # Add engineered features
    df['errorOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig'] - df['amount']
    df['errorDest'] = df['newbalanceDest'] - df['oldbalanceDest'] - df['amount']
    df['is_anomalous'] = ((df['amount'] > 0) & 
                          (df['oldbalanceOrg'] == 0) & 
                          (df['newbalanceOrig'] == 0) & 
                          (df['oldbalanceDest'] == 0) & 
                          (df['newbalanceDest'] == 0)).astype(int)

    # Reindex to match model's trained feature names
    df = df.reindex(feature_names, axis=1, fill_value=0)

    return df

# Streamlit app
st.set_page_config(page_title="Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 Online Payment Fraud Detection")
st.write("Use the sidebar to input transaction details and check for fraud detection results.")

# Sidebar inputs
st.sidebar.header("Transaction Details")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, format="%.2f")
trans_type = st.sidebar.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]).strip().upper()
oldbalanceOrg = st.sidebar.number_input("Old Balance (Sender)", min_value=0.0, format="%.2f")
newbalanceOrig = st.sidebar.number_input("New Balance (Sender)", min_value=0.0, format="%.2f")
oldbalanceDest = st.sidebar.number_input("Old Balance (Receiver)", min_value=0.0, format="%.2f")
newbalanceDest = st.sidebar.number_input("New Balance (Receiver)", min_value=0.0, format="%.2f")

# Predict button
if st.sidebar.button("Check for Fraud"):
    # Preprocess input
    input_df = preprocess_input(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, trans_type)

    # Scale input
    input_scaled = (input_df - X_train_mean) / X_train_std

    # Convert to NumPy array to avoid feature name warning
    input_scaled_array = input_scaled.to_numpy()

    # Predict
    prediction = model.predict(input_scaled_array)[0]

    # Display result
    result = "Fraud" if prediction == 1 else "Not Fraud"
    if result == "Fraud":
        st.error(f"Prediction: {result}")
    else:
>>>>>>> 03e239070bc961e6242f2374ac6f8c454fabe80a
        st.success(f"Prediction: {result}")