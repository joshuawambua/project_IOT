import joblib
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# Load model
model = joblib.load('pm25_model.pkl')
scaler = joblib.load('pm25_scaler.pkl')

print("PM2.5 Predictor Loaded")
print("=" * 40)

# Firebase setup
cred = credentials.Certificate('ecosphere-434df-firebase-adminsdk-fbsvc-02667124cf.json')
firebase_admin.initialize_app(cred, {'databaseURL': 'https://ecosphere-434df-default-rtdb.firebaseio.com/'})

def predict_pm25(temperature, humidity, pm10, mq2_adc, mq7_adc, mq8_adc):
    """Predict PM2.5 value from sensor readings"""

    features = np.array([[temperature, humidity, pm10, mq2_adc, mq7_adc, mq8_adc]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    return float(prediction)

# Test with current sensor data
ref = db.reference('air_quality/live')
sensor_data = ref.get()

if sensor_data:
    pm25_predicted = predict_pm25(
        temperature=sensor_data.get('temperature', 25),
        humidity=sensor_data.get('humidity', 50),
        pm10=sensor_data.get('pm10', 50),
        mq2_adc=sensor_data.get('mq2_adc', 500),
        mq7_adc=sensor_data.get('mq7_adc', 500),
        mq8_adc=sensor_data.get('mq8_adc', 500)
    )

    print(f"\nCurrent Sensor Readings:")
    print(f"  Temperature: {sensor_data.get('temperature')} C")
    print(f"  Humidity: {sensor_data.get('humidity')} %")
    print(f"  PM10: {sensor_data.get('pm10')} µg/m³")
    print(f"  MQ7: {sensor_data.get('mq7_adc')} ADC")

    print(f"\nXGBoost Prediction:")
    print(f"  PM2.5: {pm25_predicted:.1f} µg/m³")

    actual_pm25 = sensor_data.get('pm25', 0)
    print(f"  Actual PM2.5: {actual_pm25:.1f} µg/m³")
    print(f"  Error: {abs(pm25_predicted - actual_pm25):.1f} µg/m³")
