#!/usr/bin/env python3
"""
XGBoost PM2.5 Predictor Service
Reads live sensor data from Firebase and saves predictions
"""

import joblib
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import time
import signal
import sys
import os
import json
from datetime import datetime

# ========== CONFIGURATION ==========
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyDtRt9ojUqjJX9ItczqH8ZaouFsSCwyhCQ",
    "authDomain": "ecosphere-434df.firebaseapp.com",
    "databaseURL": "https://ecosphere-434df-default-rtdb.firebaseio.com",
    "storageBucket": "ecosphere-434df.appspot.com"
}

# Paths
SERVICE_ACCOUNT = "ecosphere-434df-firebase-adminsdk-fbsvc-02667124cf.json"
MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "pm25_scaler.pkl"
FEATURES_PATH = "xgboost_features.json"

# ========== FEATURE ENGINEERING FUNCTION ==========
def create_features_from_live_data(sensor_data, history_data=None):
    """
    Create all 38 features required by the XGBoost model
    from the live sensor data
    """

    features = {}

    # Basic sensor readings
    features['temperature'] = sensor_data.get('temperature', 25.0)
    features['humidity'] = sensor_data.get('humidity', 50.0)
    features['pm25'] = sensor_data.get('pm25', 0)
    features['pm10'] = sensor_data.get('pm10', 0)
    features['co'] = sensor_data.get('mq7_adc', 0)
    features['lpg'] = sensor_data.get('mq2_adc', 0)
    features['hydrogen'] = sensor_data.get('mq8_adc', 0)

    # Time features
    now = datetime.now()
    features['hour'] = now.hour
    features['hour_of_day'] = now.hour
    features['day_of_week'] = now.weekday()
    features['is_weekend'] = 1 if now.weekday() >= 5 else 0
    features['time_of_day'] = now.hour + now.minute/60 + now.second/3600

    # Cyclical time features
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['month_sin'] = np.sin(2 * np.pi * now.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * now.month / 12)

    # If we have history, use it for lag features
    # Otherwise use current values as fallback
    if history_data and len(history_data) >= 10:
        # Lag features from history
        features['pm25_lag1'] = history_data[-1].get('pm25', features['pm25']) if len(history_data) >= 1 else features['pm25']
        features['pm25_lag2'] = history_data[-2].get('pm25', features['pm25']) if len(history_data) >= 2 else features['pm25']
        features['pm25_lag3'] = history_data[-3].get('pm25', features['pm25']) if len(history_data) >= 3 else features['pm25']
        features['pm25_lag5'] = history_data[-5].get('pm25', features['pm25']) if len(history_data) >= 5 else features['pm25']
        features['pm25_lag10'] = history_data[-10].get('pm25', features['pm25']) if len(history_data) >= 10 else features['pm25']

        features['pm10_lag1'] = history_data[-1].get('pm10', features['pm10']) if len(history_data) >= 1 else features['pm10']
        features['pm10_lag2'] = history_data[-2].get('pm10', features['pm10']) if len(history_data) >= 2 else features['pm10']
        features['pm10_lag3'] = history_data[-3].get('pm10', features['pm10']) if len(history_data) >= 3 else features['pm10']

        features['co_lag1'] = history_data[-1].get('mq7_adc', features['co']) if len(history_data) >= 1 else features['co']
        features['co_lag2'] = history_data[-2].get('mq7_adc', features['co']) if len(history_data) >= 2 else features['co']
        features['co_lag3'] = history_data[-3].get('mq7_adc', features['co']) if len(history_data) >= 3 else features['co']

        features['lpg_lag1'] = history_data[-1].get('mq2_adc', features['lpg']) if len(history_data) >= 1 else features['lpg']
        features['lpg_lag2'] = history_data[-2].get('mq2_adc', features['lpg']) if len(history_data) >= 2 else features['lpg']

        features['hydrogen_lag1'] = history_data[-1].get('mq8_adc', features['hydrogen']) if len(history_data) >= 1 else features['hydrogen']

        features['temp_lag1'] = history_data[-1].get('temperature', features['temperature']) if len(history_data) >= 1 else features['temperature']
        features['humidity_lag1'] = history_data[-1].get('humidity', features['humidity']) if len(history_data) >= 1 else features['humidity']
    else:
        # Use current values as fallback for lag features
        features['pm25_lag1'] = features['pm25']
        features['pm25_lag2'] = features['pm25']
        features['pm25_lag3'] = features['pm25']
        features['pm25_lag5'] = features['pm25']
        features['pm25_lag10'] = features['pm25']
        features['pm10_lag1'] = features['pm10']
        features['pm10_lag2'] = features['pm10']
        features['pm10_lag3'] = features['pm10']
        features['co_lag1'] = features['co']
        features['co_lag2'] = features['co']
        features['co_lag3'] = features['co']
        features['lpg_lag1'] = features['lpg']
        features['lpg_lag2'] = features['lpg']
        features['hydrogen_lag1'] = features['hydrogen']
        features['temp_lag1'] = features['temperature']
        features['humidity_lag1'] = features['humidity']

    # Rolling statistics
    if history_data and len(history_data) >= 10:
        pm25_history = [h.get('pm25', features['pm25']) for h in history_data[-10:]]
        pm10_history = [h.get('pm10', features['pm10']) for h in history_data[-10:]]
        temp_history = [h.get('temperature', features['temperature']) for h in history_data[-10:]]
        hum_history = [h.get('humidity', features['humidity']) for h in history_data[-10:]]

        features['pm25_rolling_mean_5'] = np.mean(pm25_history[-5:]) if len(pm25_history) >= 5 else features['pm25']
        features['pm25_rolling_mean_10'] = np.mean(pm25_history) if len(pm25_history) >= 10 else features['pm25']
        features['pm10_rolling_mean_5'] = np.mean(pm10_history[-5:]) if len(pm10_history) >= 5 else features['pm10']
        features['pm10_rolling_mean_10'] = np.mean(pm10_history) if len(pm10_history) >= 10 else features['pm10']
        features['temp_rolling_mean_5'] = np.mean(temp_history[-5:]) if len(temp_history) >= 5 else features['temperature']
        features['humidity_rolling_mean_5'] = np.mean(hum_history[-5:]) if len(hum_history) >= 5 else features['humidity']
    else:
        features['pm25_rolling_mean_5'] = features['pm25']
        features['pm25_rolling_mean_10'] = features['pm25']
        features['pm10_rolling_mean_5'] = features['pm10']
        features['pm10_rolling_mean_10'] = features['pm10']
        features['temp_rolling_mean_5'] = features['temperature']
        features['humidity_rolling_mean_5'] = features['humidity']

    # Rate of change features
    features['pm25_rate'] = features['pm25'] - features['pm25_lag1']
    features['co_rate'] = features['co'] - features['co_lag1']
    features['pm25_acceleration'] = features['pm25_rate'] - (features['pm25_lag1'] - features['pm25_lag2'])

    # Interaction features
    features['temp_humidity'] = features['temperature'] * features['humidity'] / 100
    features['temp_pm25_interaction'] = features['temperature'] * features['pm25_lag1']
    features['humidity_pm25_interaction'] = features['humidity'] * features['pm25_lag1']

    return features

# ========== INITIALIZE FIREBASE ==========
cred = credentials.Certificate(SERVICE_ACCOUNT)
firebase_admin.initialize_app(cred, {
    'databaseURL': FIREBASE_CONFIG['databaseURL']
})

ref = db.reference('/')

# ========== LOAD MODEL ==========
print("Loading XGBoost model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# Load scaler if exists
scaler = None
try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded successfully!")
except:
    print("No scaler found, using raw values")

# Load feature names
try:
    with open(FEATURES_PATH, 'r') as f:
        feature_names = json.load(f)
    print(f"Model expects {len(feature_names)} features")
except:
    print("Features file not found, using default feature order")
    feature_names = None

# ========== HISTORY BUFFER ==========
history_buffer = []
MAX_HISTORY = 50

# ========== PREDICTION FUNCTION ==========
def predict_pm25(features):
    """Make prediction using loaded model"""
    try:
        # Create feature vector in correct order
        if feature_names:
            feature_vector = []
            for feature in feature_names:
                if feature in features:
                    feature_vector.append(features[feature])
                else:
                    feature_vector.append(0)
                    print(f"Warning: Missing feature {feature}")
        else:
            feature_vector = list(features.values())

        X = np.array(feature_vector).reshape(1, -1)

        # Scale if scaler exists
        if scaler:
            X = scaler.transform(X)

        prediction = model.predict(X)

        # For multi-output model, extract PM2.5 (first target)
        if isinstance(prediction, np.ndarray) and prediction.ndim > 1:
            prediction = prediction[0][0]  # First sample, first target (PM2.5)
        elif isinstance(prediction, np.ndarray):
            prediction = prediction[0]

        return max(0, float(prediction))  # Ensure non-negative
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def save_prediction_to_firebase(predicted, actual, timestamp):
    """Save prediction to Firebase"""
    error = abs(predicted - actual) if actual else 0
    error_percent = (error / (actual + 0.01)) * 100

    prediction_data = {
        'predicted': round(predicted, 2),
        'actual': round(actual, 2),
        'error': round(error, 2),
        'error_percent': round(error_percent, 2),
        'timestamp': timestamp,
        'model': 'XGBoost'
    }

    # Save to latest
    ref.child('pm25_predictions/xgboost/latest').set(prediction_data)

    # Save to history with timestamp key
    history_ref = ref.child(f'pm25_predictions/xgboost/history/{int(timestamp * 1000)}')
    history_ref.set(prediction_data)

    print(f"✅ Saved prediction: {predicted:.2f} (actual: {actual:.2f}, error: {error:.2f})")
    return True

# ========== MAIN LOOP ==========
def main():
    print("\n" + "="*50)
    print("XGBoost PM2.5 Predictor Service")
    print("="*50)
    print(f"Monitoring: /air_quality/live")
    print(f"Saving to: /pm25_predictions/xgboost/")
    print("="*50 + "\n")

    last_prediction_time = 0
    prediction_interval = 30  # seconds between predictions

    def signal_handler(sig, frame):
        print("\n🛑 Stopping predictor service...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        try:
            current_time = time.time()

            # Get latest sensor data
            live_data = ref.child('air_quality/live').get()

            if live_data and live_data.get('timestamp'):
                timestamp = live_data.get('timestamp')
                actual_pm25 = live_data.get('pm25', 0)

                # Add to history buffer
                history_buffer.append({
                    'pm25': live_data.get('pm25', 0),
                    'pm10': live_data.get('pm10', 0),
                    'mq2_adc': live_data.get('mq2_adc', 0),
                    'mq7_adc': live_data.get('mq7_adc', 0),
                    'mq8_adc': live_data.get('mq8_adc', 0),
                    'temperature': live_data.get('temperature', 25),
                    'humidity': live_data.get('humidity', 50)
                })

                # Keep only last 50 entries
                if len(history_buffer) > MAX_HISTORY:
                    history_buffer.pop(0)

                # Only make prediction if we have valid PM2.5 and enough time has passed
                if actual_pm25 > 0 and (current_time - last_prediction_time) >= prediction_interval:

                    print(f"\n📊 New sensor data at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
                    print(f"   Actual PM2.5: {actual_pm25:.2f}")

                    # Create all 38 features
                    features = create_features_from_live_data(live_data, history_buffer[:-1])

                    # Make prediction
                    predicted = predict_pm25(features)

                    if predicted is not None:
                        print(f"   Predicted PM2.5: {predicted:.2f}")
                        print(f"   Error: {abs(predicted - actual_pm25):.2f}")

                        # Save to Firebase
                        save_prediction_to_firebase(predicted, actual_pm25, timestamp)
                        last_prediction_time = current_time
                    else:
                        print("   ⚠️ Prediction failed")

                elif actual_pm25 == 0:
                    print("⏳ Waiting for valid PM2.5 readings...")

            # Wait before next check
            time.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            print("\n🛑 Stopping predictor service...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
