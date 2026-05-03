#!/usr/bin/env python3
"""
STEP 3: Decision Tree Model for Multi-Target Prediction
Predicts: PM2.5, PM10, CO, LPG, Hydrogen using cleaned data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime

print("="*60)
print("DECISION TREE MODEL FOR AIR QUALITY PREDICTION")
print("="*60)

# Load cleaned data - FIXED FILENAME
print("\nLoading cleaned data...")
df = pd.read_csv('aq_highres_clean_ml.csv')
df['reading_time'] = pd.to_datetime(df['reading_time'])

print(f"Loaded {len(df):,} records")
print(f"Date range: {df['reading_time'].min()} to {df['reading_time'].max()}")

# ============================================
# CREATE FEATURES
# ============================================
print("\nCreating features...")

# Time features
df['hour'] = df['reading_time'].dt.hour
df['day_of_week'] = df['reading_time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Lag features (previous readings - essential for prediction)
df['pm25_lag1'] = df['pm25'].shift(1)
df['pm25_lag2'] = df['pm25'].shift(2)
df['pm25_lag3'] = df['pm25'].shift(3)
df['pm25_lag5'] = df['pm25'].shift(5)
df['pm25_lag10'] = df['pm25'].shift(10)

df['pm10_lag1'] = df['pm10'].shift(1)
df['pm10_lag2'] = df['pm10'].shift(2)
df['pm10_lag3'] = df['pm10'].shift(3)

df['co_lag1'] = df['co'].shift(1)
df['co_lag2'] = df['co'].shift(2)
df['co_lag3'] = df['co'].shift(3)

df['lpg_lag1'] = df['lpg'].shift(1)
df['lpg_lag2'] = df['lpg'].shift(2)

df['hydrogen_lag1'] = df['hydrogen'].shift(1)

# Environment lags
df['temp_lag1'] = df['temperature'].shift(1)
df['humidity_lag1'] = df['humidity'].shift(1)

# Rolling statistics
df['pm25_rolling_mean_5'] = df['pm25'].rolling(window=5).mean()
df['pm25_rolling_mean_10'] = df['pm25'].rolling(window=10).mean()
df['pm10_rolling_mean_5'] = df['pm10'].rolling(window=5).mean()
df['pm10_rolling_mean_10'] = df['pm10'].rolling(window=10).mean()

# Rate of change
df['pm25_rate'] = df['pm25'] - df['pm25_lag1']
df['co_rate'] = df['co'] - df['co_lag1']

# Interaction features
df['temp_humidity'] = df['temperature'] * df['humidity'] / 100

# ============================================
# DROP NULLS (from lag features)
# ============================================
print(f"\nBefore dropping nulls: {len(df):,}")
df_clean = df.dropna()
print(f"After dropping nulls: {len(df_clean):,}")

# ============================================
# DEFINE FEATURES AND TARGETS
# ============================================

feature_columns = [
    'temperature', 'humidity',
    'hour', 'day_of_week', 'is_weekend',
    'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_lag5', 'pm25_lag10',
    'pm10_lag1', 'pm10_lag2', 'pm10_lag3',
    'co_lag1', 'co_lag2', 'co_lag3',
    'lpg_lag1', 'lpg_lag2',
    'hydrogen_lag1',
    'temp_lag1', 'humidity_lag1',
    'pm25_rolling_mean_5', 'pm25_rolling_mean_10',
    'pm10_rolling_mean_5', 'pm10_rolling_mean_10',
    'pm25_rate', 'co_rate',
    'temp_humidity'
]

target_columns = ['pm25', 'pm10', 'co', 'lpg', 'hydrogen']

X = df_clean[feature_columns]
y = df_clean[target_columns]

print(f"\nFeature matrix: {X.shape}")
print(f"Target matrix: {y.shape}")
print(f"\nFeatures ({len(feature_columns)}):")
for i, f in enumerate(feature_columns, 1):
    print(f"   {i:2d}. {f}")

# ============================================
# TRAIN/TEST SPLIT (Chronological - 80/20)
# ============================================
print("\n" + "="*60)
print("TRAIN/TEST SPLIT (Chronological)")
print("="*60)

split_idx = int(len(df_clean) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Training set: {len(X_train):,} records ({len(X_train)/len(df_clean)*100:.1f}%)")
print(f"Testing set: {len(X_test):,} records ({len(X_test)/len(df_clean)*100:.1f}%)")

# Show date ranges
train_start = df_clean['reading_time'].iloc[0]
train_end = df_clean['reading_time'].iloc[split_idx - 1]
test_start = df_clean['reading_time'].iloc[split_idx]
test_end = df_clean['reading_time'].iloc[-1]

print(f"\nTraining period: {train_start.date()} to {train_end.date()}")
print(f"Testing period: {test_start.date()} to {test_end.date()}")

# ============================================
# TRAIN DECISION TREE MODEL
# ============================================
print("\n" + "="*60)
print("Training Multi-Output Decision Tree")
print("="*60)

# Create base decision tree
base_dt = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Wrap for multi-output
model = MultiOutputRegressor(base_dt)
model.fit(X_train, y_train)

print("Model training complete!")

# ============================================
# MAKE PREDICTIONS
# ============================================
print("\nMaking predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ============================================
# EVALUATE MODEL
# ============================================
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

results = {}
for i, target in enumerate(target_columns):
    train_mae = mean_absolute_error(y_train.iloc[:, i], y_pred_train[:, i])
    train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, i], y_pred_train[:, i]))
    train_r2 = r2_score(y_train.iloc[:, i], y_pred_train[:, i])

    test_mae = mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])
    test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i]))
    test_r2 = r2_score(y_test.iloc[:, i], y_pred_test[:, i])

    results[target] = {
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'train_r2': float(train_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2)
    }

    print(f"\nTarget {target.upper()}:")
    print(f"   Training: MAE={train_mae:.2f}, RMSE={train_rmse:.2f}, R2={train_r2:.4f}")
    print(f"   Testing:  MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, R2={test_r2:.4f}")

# ============================================
# SAVE MODEL
# ============================================
print("\nSaving model...")
joblib.dump(model, 'decision_tree_model.pkl')
print("Saved: decision_tree_model.pkl")

# Save feature names
with open('features.json', 'w') as f:
    json.dump(feature_columns, f)

# Save metrics
with open('model_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: model_metrics.json")

# ============================================
# VISUALIZATIONS
# ============================================
print("\nGenerating visualizations...")

# 1. Actual vs Predicted for each target
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, target in enumerate(target_columns):
    sample_size = min(500, len(y_test))
    axes[i].scatter(y_test.iloc[:sample_size, i], y_pred_test[:sample_size, i],
                    alpha=0.3, s=10, color='#667eea')

    min_val = min(y_test.iloc[:, i].min(), y_pred_test[:, i].min())
    max_val = max(y_test.iloc[:, i].max(), y_pred_test[:, i].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    axes[i].set_xlabel(f'Actual {target.upper()}')
    axes[i].set_ylabel(f'Predicted {target.upper()}')
    axes[i].set_title(f'{target.upper()} - R2={results[target]["test_r2"]:.3f}')
    axes[i].grid(True, alpha=0.3)

axes[5].axis('off')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
print("Saved: actual_vs_predicted.png")

# 2. Time series comparison
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, target in enumerate(target_columns):
    sample_size = min(200, len(y_test))
    axes[i].plot(y_test.iloc[:sample_size, i].values, label='Actual',
                 color='#4299e1', linewidth=1)
    axes[i].plot(y_pred_test[:sample_size, i], label='Predicted',
                 color='#e53e3e', linewidth=1, linestyle='--')
    axes[i].set_title(f'{target.upper()} - Test Set Predictions')
    axes[i].set_xlabel('Sample')
    axes[i].set_ylabel('Value')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

axes[5].axis('off')
plt.tight_layout()
plt.savefig('timeseries_predictions.png', dpi=150)
print("Saved: timeseries_predictions.png")

# 3. Error distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, target in enumerate(target_columns):
    errors = y_test.iloc[:, i].values - y_pred_test[:, i]
    axes[i].hist(errors, bins=50, color='#38a169', alpha=0.7, edgecolor='black')
    axes[i].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[i].set_title(f'{target.upper()} - Error Distribution')
    axes[i].set_xlabel('Prediction Error')
    axes[i].set_ylabel('Frequency')
    axes[i].text(0.95, 0.95, f'MAE: {results[target]["test_mae"]:.2f}',
                 transform=axes[i].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

axes[5].axis('off')
plt.tight_layout()
plt.savefig('error_distributions.png', dpi=150)
print("Saved: error_distributions.png")

# 4. Performance bar chart
fig, ax = plt.subplots(figsize=(10, 6))
targets = [t.upper() for t in target_columns]
mae_values = [results[t]['test_mae'] for t in target_columns]
rmse_values = [results[t]['test_rmse'] for t in target_columns]

x = np.arange(len(targets))
width = 0.35

bars1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='#667eea')
bars2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', color='#e53e3e')

ax.set_xlabel('Target Variable')
ax.set_ylabel('Error')
ax.set_title('Model Performance by Target')
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend()
ax.grid(True, alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=150)
print("Saved: performance_comparison.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("DECISION TREE MODEL COMPLETE!")
print("="*60)

print("\nMODEL PERFORMANCE SUMMARY:")
print("-" * 50)
print(f"{'Target':<12} {'MAE':<12} {'RMSE':<12} {'R2':<12}")
print("-" * 50)
for target in target_columns:
    print(f"{target.upper():<12} {results[target]['test_mae']:<12.2f} "
          f"{results[target]['test_rmse']:<12.2f} {results[target]['test_r2']:<12.4f}")
print("-" * 50)

print("\nFiles saved:")
print("   - decision_tree_model.pkl     - Trained model")
print("   - features.json               - Feature names")
print("   - model_metrics.json          - Performance metrics")
print("   - actual_vs_predicted.png     - Scatter plots")
print("   - timeseries_predictions.png  - Time series comparison")
print("   - error_distributions.png     - Error histograms")
print("   - performance_comparison.png  - Bar chart")

print("\n" + "="*60)
