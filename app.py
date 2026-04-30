import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.title("🚗 Car Price Prediction App")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
    df['Seller_Type'] = df['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
    df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})

    df = df.dropna()

    if 'initial_price' not in df.columns:
        df['initial_price'] = df['Present_Price']

    # Features
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    model = RandomForestRegressor()
    model.fit(X, y)

    # Inputs
    year = st.number_input("Year", 2000, 2025)
    present_price = st.number_input("Present Price", 0.0)
    initial_price = st.number_input("Initial Price", 0.0)
    kms_driven = st.number_input("Kms Driven", 0)

    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.number_input("Owner", 0)

    # Convert
    fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_map = {"Dealer": 0, "Individual": 1}
    trans_map = {"Manual": 0, "Automatic": 1}

    fuel_type = fuel_map[fuel_type]
    seller_type = seller_map[seller_type]
    transmission = trans_map[transmission]

    if st.button("Predict Price"):
        input_data = [[
            year,
            present_price,
            kms_driven,
            fuel_type,
            seller_type,
            transmission,
            owner,
            initial_price
        ]]

        prediction = model.predict(input_data)
        st.success(f"Estimated Price: ₹ {prediction[0]:.2f} Lakhs")

else:
    st.warning("Please upload a CSV file")
