import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load trained XGBoost model
MODEL_PATH = "Credit_Card_Fraud_Detection.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Mappings used during training for LabelEncoder
CATEGORY_MAPPING = {
    "entertainment": 0,
    "food_dining": 1,
    "gas_transport": 2,
    "grocery_net": 3,
    "grocery_pos": 4,
    "health_fitness": 5,
    "home": 6,
    "kids_pets": 7,
    "misc_net": 8,
    "misc_pos": 9,
    "personal_care": 10,
    "shopping_net": 11,
    "shopping_pos": 12,
    "travel": 13
}

GENDER_MAPPING = {
    "F": 0,
    "M": 1
}

def calculate_haversine(lat1, lon1, lat2, lon2):
    """Calculates distance between user and merchant in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def preprocess_input(data):
    """
    Transforms raw user inputs into the feature array required by the model.
    """
    # Parse dates
    trans_dt = datetime.strptime(data['trans_date_trans_time'], "%Y-%m-%dT%H:%M")
    dob_dt = datetime.strptime(data['dob'], "%Y-%m-%d")

    # Time/Date extractions
    trans_hour = trans_dt.hour
    trans_day = trans_dt.day
    trans_month = trans_dt.month
    age = (trans_dt - dob_dt).days // 365

    # Coordinates & Distance calculation
    user_lat = float(data['lat'])
    user_long = float(data['long'])
    merch_lat = float(data['merch_lat'])
    merch_long = float(data['merch_long'])
    
    distance_km = calculate_haversine(user_lat, user_long, merch_lat, merch_long)

    # Encode categorical variables
    category_encoded = CATEGORY_MAPPING.get(data['category'], 0)
    gender_encoded = GENDER_MAPPING.get(data['gender'], 0)
    amt = float(data['amt'])

    # Construct feature dataframe in exact column order expected by model
    features_df = pd.DataFrame([{
        "category": category_encoded,
        "amt": amt,
        "gender": gender_encoded,
        "lat": user_lat,
        "long": user_long,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "trans_hour": trans_hour,
        "trans_day": trans_day,
        "trans_month": trans_month,
        "age": age,
        "distance_km": distance_km
    }])

    return features_df

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly."}), 500

    try:
        # Check if request is JSON or Form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        features = preprocess_input(data)
        
        # Prediction
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        result = {
            "is_fraud": bool(prediction),
            "label": "Fraudulent" if prediction == 1 else "Legitimate",
            "fraud_probability": round(probability * 100, 2)
        }

        if request.is_json:
            return jsonify(result)
        
        return render_template("index.html", result=result, input_data=data)

    except Exception as e:
        return render_template("index.html", error=str(e))


# RUN
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
