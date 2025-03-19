import streamlit as st
import joblib
import numpy as np
import os

# ✅ Define Absolute Paths
MODEL_PATH = "/Users/ngoubimaximilliandiamgha/Desktop/PythonProject6/xgboost_housing_model.pkl"
SCALER_PATH = "/Users/ngoubimaximilliandiamgha/Desktop/PythonProject6/scaler.pkl"

st.title("🏠 House Price Prediction App")

# ✅ Check if Model Exists Before Loading
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or Scaler file not found! Please retrain the model.")
    st.stop()
else:
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        st.success("✅ Model and Scaler loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model/scaler: {e}")
        st.stop()

# ✅ Input Fields
ZHVI_Growth = st.number_input("ZHVI Growth (%)", value=3.2)
ZORI_Growth = st.number_input("ZORI Growth (%)", value=1.5)
Sales_Growth = st.number_input("Sales Growth (%)", value=2.0)
Affordability_Index = st.number_input("Affordability Index", value=0.85)
Supply_Demand_Ratio = st.number_input("Supply Demand Ratio", value=1.2)

# ✅ Predict Button
if st.button("Predict"):
    try:
        input_data = np.array([[ZHVI_Growth, ZORI_Growth, Sales_Growth, Affordability_Index, Supply_Demand_Ratio]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        st.success(f"🏡 Predicted House Value: **${round(prediction, 2)}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
