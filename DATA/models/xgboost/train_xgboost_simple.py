#!/usr/bin/env python3
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading data...")
df = pd.read_csv('aq_highres_clean_ml.csv')
df['reading_time'] = pd.to_datetime(df['reading_time'])
df = df.sort_values('reading_time')

# Use ONLY features available from live sensor
features = ['temperature', 'humidity', 'pm10', 'mq2_adc', 'mq7_adc', 'mq8_adc']
target = 'pm25'

# Add simple lag features
df['pm25_prev'] = df['pm25'].shift(1)
df['pm25_prev2'] = df['pm25'].shift(2)
df['temperature_prev'] = df['temperature'].shift(1)

# Final feature set
final_features = features + ['pm25_prev', 'pm25_prev2', 'temperature_prev']

# Drop NaN
df_clean = df.dropna()
print(f"Clean data: {len(df_clean)} records")

# Split
split = int(len(df_clean) * 0.8)
X = df_clean[final_features]
y = df_clean[target]

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train model
print("Training Random Forest...")
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResults:")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.4f}")

# Save
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(final_features, 'xgboost_features.pkl')
print("\n✅ Model saved!")
