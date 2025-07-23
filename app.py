import streamlit as st
import pandas as pd
import joblib

# Load trained model and pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")
threshold = 0.7  # Use the same threshold you used in training

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Online Payment Fraud Detection")

st.markdown("""
Enter the transaction details below to check if the transaction is likely to be **fraudulent** or **legitimate**.
""")

# Input fields
type_input = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0)

if st.button("🔍 Predict Fraud"):
    # Feature engineering
    errorOrig = oldbalanceOrg - amount - newbalanceOrig
    errorDest = newbalanceDest + amount - oldbalanceDest

    # Create DataFrame
    input_data = pd.DataFrame([{
        "type": type_input,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "errorOrig": errorOrig,
        "errorDest": errorDest
    }])

    # Preprocess input and predict
    input_transformed = pipeline.transform(input_data)
    prob = model.predict_proba(input_transformed)[:, 1][0]
    prediction = int(prob >= threshold)

    # Display result
    st.subheader("🔎 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Transaction is Legitimate. (Probability: {prob:.2f})")
