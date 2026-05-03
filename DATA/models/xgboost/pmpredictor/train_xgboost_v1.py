#!/usr/bin/env python3
"""
XGBoost PM2.5 Training Script v1.0
Trains model using historical air quality data
"""

import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("="*60)
print("XGBoost PM2.5 Training v1.0")
print("="*60)

# Load data
print("\n[1/5] Loading data...")
df = pd.read_csv('aq_highres_clean_ml.csv')
df['reading_time'] = pd.to_datetime(df['reading_time'])
df = df.sort_values('reading_time')
print(f"     Loaded {len(df):,} records")

# Feature engineering
print("\n[2/5] Creating features...")

# Base features from sensors
features = ['temperature', 'humidity', 'pm10', 'mq2_adc', 'mq7_adc', 'mq8_adc']
target = 'pm25'

# Lag features
df['pm25_lag1'] = df['pm25'].shift(1)
df['pm25_lag2'] = df['pm25'].shift(2)
df['pm25_lag3'] = df['pm25'].shift(3)
df['pm10_lag1'] = df['pm10'].shift(1)
df['temp_lag1'] = df['temperature'].shift(1)
df['hum_lag1'] = df['humidity'].shift(1)

# Rolling averages
df['pm25_roll5'] = df['pm25'].rolling(window=5).mean()
df['pm25_roll10'] = df['pm25'].rolling(window=10).mean()

# Rate of change
df['pm25_rate'] = df['pm25'] - df['pm25_lag1']

# Time features
df['hour'] = df['reading_time'].dt.hour
df['day_of_week'] = df['reading_time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Final feature list
final_features = features + [
    'pm25_lag1', 'pm25_lag2', 'pm25_lag3',
    'pm10_lag1', 'temp_lag1', 'hum_lag1',
    'pm25_roll5', 'pm25_roll10',
    'pm25_rate', 'hour', 'day_of_week', 'is_weekend'
]

print(f"     Created {len(final_features)} features")

# Clean data
print("\n[3/5] Cleaning data...")
df_clean = df.dropna()
print(f"     Clean records: {len(df_clean):,}")

# Split data
print("\n[4/5] Splitting data...")
X = df_clean[final_features]
y = df_clean[target]

split_idx = int(len(df_clean) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"     Training: {len(X_train):,} records")
print(f"     Testing: {len(X_test):,} records")

# Train model
print("\n[5/5] Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)
print(f"Training:   MAE={train_mae:.2f}, RMSE={train_rmse:.2f}, R2={train_r2:.4f}")
print(f"Testing:    MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, R2={test_r2:.4f}")

# Save artifacts
print("\n💾 Saving artifacts...")
joblib.dump(model, 'xgboost_pm25_model_v1.pkl')
joblib.dump(final_features, 'xgboost_features_v1.pkl')
print("     ✓ xgboost_pm25_model_v1.pkl")
print("     ✓ xgboost_features_v1.pkl")

# Save metrics
metrics = {
    'model_version': '1.0',
    'train_mae': float(train_mae),
    'train_rmse': float(train_rmse),
    'train_r2': float(train_r2),
    'test_mae': float(test_mae),
    'test_rmse': float(test_rmse),
    'test_r2': float(test_r2),
    'n_features': len(final_features),
    'features': final_features
}

with open('model_metrics_v1.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("     ✓ model_metrics_v1.json")

print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)
