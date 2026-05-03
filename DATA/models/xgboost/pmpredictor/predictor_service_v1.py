#!/usr/bin/env python3
"""
XGBoost PM2.5 Predictor Service v1.0 - FIXED JSON Serialization
Reads live sensor data and saves predictions to Firebase
"""

import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np
import time
import signal
import sys
import os
from datetime import datetime

# Convert numpy types to Python types
def convert_to_serializable(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj

print("="*60)
print("XGBoost PM2.5 Predictor Service v1.0")
print("="*60)

# Load model and features
print("\n[1/4] Loading model...")
try:
    model = joblib.load('xgboost_pm25_model_v1.pkl')
    print("     ✓ Model loaded")
except Exception as e:
    print(f"     ✗ Error loading model: {e}")
    sys.exit(1)

try:
    features = joblib.load('xgboost_features_v1.pkl')
    print(f"     ✓ Features loaded ({len(features)} features)")
except Exception as e:
    print(f"     ✗ Error loading features: {e}")
    sys.exit(1)

# Initialize Firebase
print("\n[2/4] Initializing Firebase...")
try:
    service_account = "ecosphere-434df-firebase-adminsdk-fbsvc-02667124cf.json"
    if not os.path.exists(service_account):
        print(f"     ✗ Service account file not found: {service_account}")
        print("     Make sure you're in the correct directory")
        sys.exit(1)

    cred = credentials.Certificate(service_account)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://ecosphere-434df-default-rtdb.firebaseio.com/'
    })
    print("     ✓ Firebase initialized")
except Exception as e:
    print(f"     ✗ Firebase error: {e}")
    sys.exit(1)

# History buffer
history_buffer = []
MAX_HISTORY = 20
prediction_count = 0

def build_features(live_data):
    """Build feature vector from live data and history"""
    try:
        # Current values
        temp = float(live_data.get('temperature', 25.0))
        hum = float(live_data.get('humidity', 50.0))
        pm10 = float(live_data.get('pm10', 0))
        mq2 = float(live_data.get('mq2_adc', 0))
        mq7 = float(live_data.get('mq7_adc', 0))
        mq8 = float(live_data.get('mq8_adc', 0))

        # Default lag values
        pm25_lag1 = pm25_lag2 = pm25_lag3 = 0.0
        pm10_lag1 = 0.0
        temp_lag1 = temp
        hum_lag1 = hum
        pm25_roll5 = pm25_roll10 = 0.0
        pm25_rate = 0.0

        # Use history for lag features
        if len(history_buffer) >= 1:
            pm25_lag1 = float(history_buffer[-1].get('pm25', 0))
            pm10_lag1 = float(history_buffer[-1].get('pm10', 0))
            temp_lag1 = float(history_buffer[-1].get('temperature', temp))
            hum_lag1 = float(history_buffer[-1].get('humidity', hum))

        if len(history_buffer) >= 2:
            pm25_lag2 = float(history_buffer[-2].get('pm25', 0))
            pm25_rate = pm25_lag1 - pm25_lag2

        if len(history_buffer) >= 3:
            pm25_lag3 = float(history_buffer[-3].get('pm25', 0))

        # Rolling averages
        if len(history_buffer) >= 5:
            recent = [float(h.get('pm25', 0)) for h in history_buffer[-5:]]
            pm25_roll5 = sum(recent) / len(recent)

        if len(history_buffer) >= 10:
            recent = [float(h.get('pm25', 0)) for h in history_buffer[-10:]]
            pm25_roll10 = sum(recent) / len(recent)

        # Time features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        # Create feature dictionary with Python floats
        feature_dict = {
            'temperature': temp,
            'humidity': hum,
            'pm10': pm10,
            'mq2_adc': mq2,
            'mq7_adc': mq7,
            'mq8_adc': mq8,
            'pm25_lag1': pm25_lag1,
            'pm25_lag2': pm25_lag2,
            'pm25_lag3': pm25_lag3,
            'pm10_lag1': pm10_lag1,
            'temp_lag1': temp_lag1,
            'hum_lag1': hum_lag1,
            'pm25_roll5': pm25_roll5,
            'pm25_roll10': pm25_roll10,
            'pm25_rate': pm25_rate,
            'hour': float(hour),
            'day_of_week': float(day_of_week),
            'is_weekend': float(is_weekend)
        }

        # Create vector in correct order
        X = [[float(feature_dict[f]) for f in features]]
        return X

    except Exception as e:
        print(f"     Feature error: {e}")
        return None

def predict_pm25(live_data):
    """Make prediction using the model"""
    try:
        X = build_features(live_data)
        if X is None:
            return None

        prediction = model.predict(X)[0]
        # Convert to Python float and ensure reasonable range
        prediction = float(prediction)
        if prediction < 0:
            prediction = 0
        if prediction > 500:
            prediction = 500

        return prediction

    except Exception as e:
        print(f"     Prediction error: {e}")
        return None

def save_prediction(predicted, actual, timestamp):
    """Save prediction to Firebase - with proper JSON serialization"""
    global prediction_count

    # Convert everything to Python native types
    predicted = float(predicted)
    actual = float(actual)
    error = float(abs(predicted - actual))
    error_percent = float((error / (actual + 0.01)) * 100)
    timestamp = float(timestamp)

    data = {
        'predicted': round(predicted, 1),
        'actual': round(actual, 1),
        'error': round(error, 1),
        'error_percent': round(error_percent, 1),
        'timestamp': timestamp,
        'model': 'XGBoost_v1'
    }

    try:
        # Save to latest
        db.reference('pm25_predictions/latest').set(data)

        # Save to history
        date_path = datetime.fromtimestamp(timestamp).strftime('%Y/%m/%d')
        db.reference(f'pm25_predictions/history/{date_path}/{int(timestamp*1000)}').set(data)

        prediction_count += 1
        return True
    except Exception as e:
        print(f"     Firebase save error: {e}")
        return False

def main():
    global prediction_count

    print("\n[3/4] Starting predictor...")
    print(f"     Saving to: /pm25_predictions/latest")
    print(f"     Prediction interval: 30 seconds")

    def shutdown(sig, frame):
        print(f"\n[4/4] Shutting down... (Total predictions: {prediction_count})")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    last_prediction = 0
    prediction_interval = 30
    waiting_message_shown = False

    print("\n" + "="*60)
    print("SERVICE RUNNING - Waiting for sensor data...")
    print("="*60 + "\n")

    while True:
        try:
            # Get live data
            live_ref = db.reference('air_quality/live')
            live = live_ref.get()

            if live and live.get('pm25', 0) > 0:
                waiting_message_shown = False
                now = time.time()
                actual = float(live.get('pm25'))
                timestamp = float(live.get('timestamp', now))

                # Store in history
                history_buffer.append({
                    'pm25': actual,
                    'pm10': float(live.get('pm10', 0)),
                    'temperature': float(live.get('temperature', 25)),
                    'humidity': float(live.get('humidity', 50))
                })

                if len(history_buffer) > MAX_HISTORY:
                    history_buffer.pop(0)

                # Predict every 30 seconds
                if now - last_prediction >= prediction_interval:
                    predicted = predict_pm25(live)

                    if predicted is not None:
                        if save_prediction(predicted, actual, timestamp):
                            print(f"\n📊 Prediction #{prediction_count}")
                            print(f"   Time: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
                            print(f"   Actual PM2.5: {actual:.1f} µg/m³")
                            print(f"   Predicted PM2.5: {predicted:.1f} µg/m³")
                            print(f"   Error: {abs(predicted-actual):.1f} µg/m³")
                            last_prediction = now
                    else:
                        print("⚠️ Prediction failed")
            else:
                if not waiting_message_shown:
                    print("⏳ Waiting for sensor data from ESP32...")
                    waiting_message_shown = True

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
