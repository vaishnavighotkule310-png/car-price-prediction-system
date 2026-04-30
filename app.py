import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Title
st.title("🚗 Car Price Prediction App")

# Load dataset
df = pd.read_csv("cardekho.csv")

# Preprocessing
df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
df['Seller_Type'] = df['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})

df = df.dropna()

# ✅ Add initial_price if not present
if 'initial_price' not in df.columns:
    if 'Present_Price' in df.columns:
        df['initial_price'] = df['Present_Price']
    else:
        df['initial_price'] = 0

# Features & target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# User Inputs
year = st.number_input("Year", 2000, 2025)
present_price = st.number_input("Present Price", 0.0)

# ✅ New input for initial price
initial_price = st.number_input("Initial Price (Original Price)", 0.0)

kms_driven = st.number_input("Kms Driven", 0)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.number_input("Owner", 0)

# Convert inputs
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

# Predict button
if st.button("Predict Price"):
    prediction = model.predict([[year, present_price, initial_price,
