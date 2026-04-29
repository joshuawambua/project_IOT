import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import json

print("Training XGBoost REGRESSION Model (Predict PM2.5)")
print("=" * 50)

# Load data
df = pd.read_csv('aq_highres_clean_ml.csv')
print(f"Loaded {len(df)} rows")

# Features (inputs)
feature_cols = ['temperature', 'humidity', 'pm10', 'mq2_adc', 'mq7_adc', 'mq8_adc']

# Target (output) - predict PM2.5 numerical value
target_col = 'pm25'

# Remove rows with missing values
df_clean = df[feature_cols + [target_col]].dropna()
print(f"After cleaning: {len(df_clean)} rows")

# Features and target
X = df_clean[feature_cols].values
y = df_clean[target_col].values

print(f"\nPM2.5 Range: {y.min():.1f} to {y.max():.1f} µg/m³")
print(f"PM2.5 Mean: {y.mean():.1f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"\nR² Score (training): {train_score:.4f}")
print(f"R² Score (test): {test_score:.4f}")

# Calculate error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error: {mae:.2f} µg/m³")
print(f"RMSE: {rmse:.2f} µg/m³")

# Save model
joblib.dump(model, 'pm25_model.pkl')
joblib.dump(scaler, 'pm25_scaler.pkl')

# Save results
results = {
    'model_type': 'XGBoost Regressor',
    'target': 'pm25',
    'features': feature_cols,
    'r2_train': float(train_score),
    'r2_test': float(test_score),
    'mae': float(mae),
    'rmse': float(rmse),
    'pm25_range': [float(y.min()), float(y.max())],
    'pm25_mean': float(y.mean())
}

with open('pm25_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Model saved as: pm25_model.pkl")
print("✓ Scaler saved as: pm25_scaler.pkl")
print("✓ Results saved as: pm25_results.json")
