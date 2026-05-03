#!/usr/bin/env python3
import firebase_admin
from firebase_admin import credentials, db
import joblib
import time
import signal
import sys
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate("ecosphere-434df-firebase-adminsdk-fbsvc-02667124cf.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ecosphere-434df-default-rtdb.firebaseio.com/'
})

# Load model
print("Loading model...")
model = joblib.load('xgboost_model.pkl')
features = joblib.load('xgboost_features.pkl')
print(f"Model loaded. Expects {len(features)} features")

# History buffer
history = []

def get_prediction(live_data):
    """Extract features and predict"""
    try:
        # Get current values
        feat_dict = {
            'temperature': live_data.get('temperature', 25.0),
            'humidity': live_data.get('humidity', 50.0),
            'pm10': live_data.get('pm10', 0),
            'mq2_adc': live_data.get('mq2_adc', 0),
            'mq7_adc': live_data.get('mq7_adc', 0),
            'mq8_adc': live_data.get('mq8_adc', 0),
        }

        # Add lag features from history
        if len(history) >= 1:
            feat_dict['pm25_prev'] = history[-1].get('pm25', 0)
            feat_dict['temperature_prev'] = history[-1].get('temperature', feat_dict['temperature'])
        else:
            feat_dict['pm25_prev'] = feat_dict['pm10']
            feat_dict['temperature_prev'] = feat_dict['temperature']

        if len(history) >= 2:
            feat_dict['pm25_prev2'] = history[-2].get('pm25', 0)
        else:
            feat_dict['pm25_prev2'] = feat_dict['pm25_prev']

        # Create feature vector in correct order
        X = [[feat_dict[f] for f in features]]

        # Predict
        prediction = model.predict(X)[0]
        return max(0, prediction)

    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def save_to_firebase(predicted, actual, timestamp):
    """Save prediction to Firebase"""
    error = abs(predicted - actual)
    error_pct = (error / actual) * 100 if actual > 0 else 0

    data = {
        'predicted': round(predicted, 1),
        'actual': round(actual, 1),
        'error': round(error, 1),
        'error_percent': round(error_pct, 1),
        'timestamp': timestamp
    }

    # Save to latest
    db.reference('pm25_predictions/xgboost/latest').set(data)

    # Save to history
    db.reference(f'pm25_predictions/xgboost/history/{int(timestamp*1000)}').set(data)

    print(f"✓ PM2.5: Predicted={predicted:.1f} Actual={actual:.1f} Error={error:.1f}")
    return True

def main():
    print("\n" + "="*50)
    print("XGBoost Predictor Running")
    print("="*50)

    def shutdown(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    last_prediction = 0

    while True:
        try:
            # Get latest sensor data
            live = db.reference('air_quality/live').get()

            if live and live.get('pm25', 0) > 0:
                now = time.time()

                # Store in history
                history.append({
                    'pm25': live.get('pm25'),
                    'temperature': live.get('temperature')
                })
                if len(history) > 10:
                    history.pop(0)

                # Predict every 30 seconds
                if now - last_prediction >= 30:
                    predicted = get_prediction(live)

                    if predicted is not None:
                        save_to_firebase(
                            predicted,
                            live.get('pm25'),
                            live.get('timestamp', now)
                        )
                        last_prediction = now

            time.sleep(5)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
