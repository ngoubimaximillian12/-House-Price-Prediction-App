from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ Define Paths for Model and Scaler
MODEL_PATH = "/Users/ngoubimaximilliandiamgha/Desktop/PythonProject6/xgboost_housing_model.pkl"
SCALER_PATH = "/Users/ngoubimaximilliandiamgha/Desktop/PythonProject6/scaler.pkl"

# ✅ Load Model & Scaler (Stop Execution if Not Found)
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")
        model, scaler = None, None  # Prevent execution if error occurs
else:
    print("❌ Model or Scaler file not found! Please retrain the model.")
    model, scaler = None, None  # Prevent execution if files are missing


@app.route('/predict', methods=['POST'])
def predict():
    """Predict house price based on input features."""
    if model is None or scaler is None:
        return jsonify({"error": "❌ Model is not available. Please retrain it."}), 500

    try:
        data = request.json
        input_data = np.array([[data['ZHVI_Growth'], data['ZORI_Growth'], data['Sales_Growth'],
                                data['Affordability_Index'], data['Supply_Demand_Ratio']]])
        input_scaled = scaler.transform(input_data)
        predicted_price = model.predict(input_scaled)[0]
        return jsonify({"Predicted Home Value": round(predicted_price, 2)})

    except Exception as e:
        return jsonify({"error": f"❌ Prediction failed: {e}"}), 400


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5001)  # ✅ Run on port 5001 to avoid conflicts
