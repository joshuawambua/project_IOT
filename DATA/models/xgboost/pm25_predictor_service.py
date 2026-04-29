

import time
import logging
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import joblib
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PREDICTION_INTERVAL = 60
FIREBASE_CRED = 'ecosphere-434df-firebase-adminsdk-fbsvc-02667124cf.json'
FIREBASE_URL = 'https://ecosphere-434df-default-rtdb.firebaseio.com/'

print("Loading PM2.5 Prediction Model...")
model = joblib.load('pm25_model.pkl')
scaler = joblib.load('pm25_scaler.pkl')
print("Model loaded")

print("Connecting to Firebase...")
cred = credentials.Certificate(FIREBASE_CRED)
firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
print("Firebase connected")

def predict_pm25(temperature, humidity, pm10, mq2_adc, mq7_adc, mq8_adc):
    features = np.array([[temperature, humidity, pm10, mq2_adc, mq7_adc, mq8_adc]])
    features_scaled = scaler.transform(features)
    return float(model.predict(features_scaled)[0])

print(f"Predicting every {PREDICTION_INTERVAL} seconds")
print("Waiting for sensor data...")

while True:
    try:
        ref = db.reference('air_quality/live')
        data = ref.get()

        if data:
            predicted = predict_pm25(
                temperature=data.get('temperature', 25),
                humidity=data.get('humidity', 50),
                pm10=data.get('pm10', 50),
                mq2_adc=data.get('mq2_adc', 500),
                mq7_adc=data.get('mq7_adc', 500),
                mq8_adc=data.get('mq8_adc', 500)
            )

            actual = data.get('pm25', 0)
            error = abs(predicted - actual)
            error_percent = (error / actual * 100) if actual > 0 else 0

            result = {
                'timestamp': datetime.now().isoformat(),
                'predicted': round(predicted, 2),
                'actual': round(actual, 2),
                'error': round(error, 2),
                'error_percent': round(error_percent, 1),
                'sensor_data': {
                    'temperature': data.get('temperature'),
                    'humidity': data.get('humidity'),
                    'pm10': data.get('pm10'),
                    'mq2_adc': data.get('mq2_adc'),
                    'mq7_adc': data.get('mq7_adc'),
                    'mq8_adc': data.get('mq8_adc')
                }
            }

            db.reference('pm25_predictions/latest').set(result)

            now = datetime.now()
            path = f"pm25_predictions/history/{now.year}/{now.month:02d}/{now.day:02d}"
            db.reference(path).push().set(result)

            stats_ref = db.reference('pm25_predictions/stats')
            stats = stats_ref.get() or {}
            total = stats.get('total', 0) + 1
            total_error = stats.get('total_error', 0) + error
            stats_ref.set({
                'total': total,
                'avg_error': round(total_error / total, 2),
                'last_prediction': result['timestamp'],
                'running': True
            })

            logger.info(f"PM2.5 - Predicted: {predicted:.1f} | Actual: {actual:.1f} | Error: {error:.1f} µg/m³ ({error_percent:.1f}%)")
        else:
            logger.warning("Waiting for sensor data...")

        time.sleep(PREDICTION_INTERVAL)

    except KeyboardInterrupt:
        print("Stopping...")
        break
    except Exception as e:
        logger.error(f"Error: {e}")
        time.sleep(5)
