import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("🚗 Car Price Prediction System")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("cardekho.csv")
    return data

data = load_data()

# Data preprocessing
data['Car_Age'] = 2025 - data['Year']
data.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Convert categorical data
data = pd.get_dummies(data, drop_first=True)

# Split features & target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# User inputs
year = st.number_input("Year of Purchase", min_value=2000, max_value=2025)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Owner", [0, 1, 2, 3])

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Convert input to model format
car_age = 2025 - year

fuel_diesel = 1 if fuel == "Diesel" else 0
fuel_petrol = 1 if fuel == "Petrol" else 0

seller_individual = 1 if seller == "Individual" else 0
transmission_manual = 1 if transmission == "Manual" else 0

# Predict
if st.button("Predict Price"):
    
    input_data = np.array([[present_price, kms_driven, owner, car_age,
                            fuel_diesel, fuel_petrol,
                            seller_individual, transmission_manual]])
    
    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹ {round(prediction[0], 2)} lakhs")
