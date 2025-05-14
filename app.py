import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load artifacts
scaler = joblib.load('scaler.pkl')
model  = joblib.load('california_housing_mlp.pkl')

st.title("🏠 California Housing Price Predictor")

# 2. Input form
st.sidebar.header("Enter house parameters")
def user_input_features():
    MedInc    = st.sidebar.slider('Median Income (10k$)', 0.5, 15.0, 5.0)
    HouseAge  = st.sidebar.slider('House Age (years)', 1, 52, 20)
    AveRooms  = st.sidebar.slider('Avg Rooms', 1.0, 10.0, 5.0)
    AveBedrms = st.sidebar.slider('Avg Bedrooms', 1.0, 5.0, 2.0)
    Population= st.sidebar.slider('Population', 100, 5000, 1000)
    AveOccup  = st.sidebar.slider('Avg Occupancy', 1.0, 10.0, 3.0)
    Latitude  = st.sidebar.slider('Latitude', 32.0, 42.0, 35.0)
    Longitude = st.sidebar.slider('Longitude', -124.0, -114.0, -119.0)
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("Input parameters")
st.write(input_df)

# 3. Preprocess & predict
X_scaled = scaler.transform(input_df)
prediction = model.predict(X_scaled)

st.subheader("Predicted Median House Value (100k$)")
st.write(f"**{prediction[0]:.2f}**")