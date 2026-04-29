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
QUALITY_LABELS = {0: "GOOD", 2: "MODERATE"}

print("XGBoost Predictor Service Starting...")
print("Loading model...")

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
print("Model loaded")

print("Connecting to Firebase...")
cred = credentials.Certificate(FIREBASE_CRED)
firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_URL})
print("Firebase connected")

print(f"Will predict every {PREDICTION_INTERVAL} seconds")
print("Waiting for sensor data...")

while True:
    try:
        # Get sensor data
        ref = db.reference('air_quality/live')
        data = ref.get()

        if data:
            # Prepare features
            features = np.array([[
                float(data.get('temperature', 25)),
                float(data.get('humidity', 50)),
                float(data.get('pm25', 35)),
                float(data.get('pm10', 45)),
                int(data.get('mq2_adc', 500)),
                int(data.get('mq7_adc', 500)),
                int(data.get('mq8_adc', 500))
            ]])

            # Predict
            scaled = scaler.transform(features)
            pred_encoded = model.predict(scaled)[0]
            pred_original = label_encoder.inverse_transform([pred_encoded])[0]
            proba = model.predict_proba(scaled)[0]
            confidence = float(max(proba))

            actual = int(data.get('quality', 0))
            match = (actual == pred_original)

            # Save to Firebase - convert boolean to string to avoid serialization error
            result = {
                'timestamp': datetime.now().isoformat(),
                'actual': {'code': actual, 'label': QUALITY_LABELS.get(actual, 'UNKNOWN')},
                'predicted': {'code': int(pred_original), 'label': QUALITY_LABELS.get(int(pred_original), 'UNKNOWN'), 'confidence': confidence},
                'match': "Yes" if match else "No"
            }

            db.reference('xgboost/predictions/latest').set(result)

            # Update stats
            now = datetime.now()
            path = f"xgboost/predictions/history/{now.year}/{now.month:02d}/{now.day:02d}"
            db.reference(path).push().set(result)

            logger.info(f"Actual: {actual} ({QUALITY_LABELS.get(actual, 'UNKNOWN')}) | Predicted: {pred_original} ({QUALITY_LABELS.get(int(pred_original), 'UNKNOWN')}) | Match: {match} | Conf: {confidence:.2%}")
        else:
            logger.warning("Waiting for sensor data...")

        time.sleep(PREDICTION_INTERVAL)

    except Exception as e:
        logger.error(f"Error: {e}")
        time.sleep(5)
