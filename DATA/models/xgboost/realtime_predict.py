import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

print("✓ Model, scaler, and label encoder loaded")
print(f"✓ Quality classes: {label_encoder.classes_}")

def predict_airquality(temperature, humidity, pm25, pm10, mq2_adc, mq7_adc, mq8_adc):
    """Predict air quality based on sensor readings"""

    # Create feature array
    features = np.array([[temperature, humidity, pm25, pm10,
                          mq2_adc, mq7_adc, mq8_adc]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict (returns encoded value)
    prediction_encoded = model.predict(features_scaled)[0]
    probabilities_encoded = model.predict_proba(features_scaled)[0]

    # Convert back to original quality_flag value
    prediction_original = label_encoder.inverse_transform([prediction_encoded])[0]

    # Map to readable labels
    quality_map = {0: 'Good', 2: 'Moderate'}  # Adjust based on your actual classes

    return {
        'quality_code': int(prediction_original),
        'quality_label': quality_map.get(int(prediction_original), f'Class {prediction_original}'),
        'confidence': float(max(probabilities_encoded)),
        'probabilities': {f'Class {label_encoder.classes_[i]}': float(p) for i, p in enumerate(probabilities_encoded)}
    }

# Test with sample data
if __name__ == "__main__":
    # Example sensor reading
    result = predict_airquality(
        temperature=25.5,
        humidity=60.0,
        pm25=35.2,
        pm10=45.1,
        mq2_adc=512,
        mq7_adc=489,
        mq8_adc=501
    )

    print("\n📊 Prediction Result:")
    print(f"   Quality Code: {result['quality_code']}")
    print(f"   Quality Label: {result['quality_label']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print("\n   Probabilities:")
    for label, prob in result['probabilities'].items():
        print(f"     {label}: {prob:.2%}")
