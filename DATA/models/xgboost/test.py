"""
Test the model before starting service
"""
import joblib
import numpy as np

print("Testing XGBoost Model...")
print("=" * 30)

# Load
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
print("✓ Model loaded")

# Test
test = np.array([[25.5, 60, 35.2, 45.1, 512, 489, 501]])
scaled = scaler.transform(test)
pred = model.predict(scaled)[0]
original = le.inverse_transform([pred])[0]
proba = model.predict_proba(scaled)[0]

print(f"\nTest Input: Temp=25.5°C, PM2.5=35.2")
print(f"Predicted: {original}")
print(f"Confidence: {max(proba):.2%}")
print("\n✅ Ready to go!")
