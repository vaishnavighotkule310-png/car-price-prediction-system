import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚗 Car Price Prediction System")

# Inputs
year = st.number_input("Year of Purchase", 2000, 2025)
km_driven = st.number_input("Kilometers Driven", 0, 500000)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", [0, 1, 2, 3])

# Convert categorical to match training
fuel_diesel = 1 if fuel == "Diesel" else 0
seller_individual = 1 if seller_type == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# Prediction
if st.button("Predict Price"):
    input_data = np.array([[year, km_driven, owner,
                            fuel_diesel,
                            seller_individual,
                            transmission_manual]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Car Price: ₹{int(prediction[0])}")
