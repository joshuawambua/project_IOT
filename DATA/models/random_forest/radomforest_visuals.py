#!/usr/bin/env python3
"""
Random Forest Model for Multi-Target Prediction
Predicts: PM2.5, PM10, CO (from MQ7), LPG (from MQ2), Hydrogen (from MQ8)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')

print("="*60)
print("RANDOM FOREST MODEL FOR AIR QUALITY PREDICTION")
print("="*60)

# Load cleaned data
print("\nLoading cleaned data...")
df = pd.read_csv('aq_highres_clean_ml.csv')
df['reading_time'] = pd.to_datetime(df['reading_time'])

print(f"Loaded {len(df):,} records")
print(f"Date range: {df['reading_time'].min()} to {df['reading_time'].max()}")

# Map sensor readings to pollutants
print("\nMapping sensor readings to pollutants...")
df['co'] = df['mq7_voltage']
df['lpg'] = df['mq2_voltage']
df['hydrogen'] = df['mq8_voltage']

# Create features
print("\nCreating features...")
df['hour_of_day'] = df['hour']
df['day_of_week'] = pd.to_datetime(df['reading_time']).dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['time_of_day'] = df['hour'] + df['minute']/60 + df['second']/3600

# Lag features
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

df['temp_lag1'] = df['temperature'].shift(1)
df['humidity_lag1'] = df['humidity'].shift(1)

# Rolling statistics
df['pm25_rolling_mean_5'] = df['pm25'].rolling(window=5).mean()
df['pm25_rolling_mean_10'] = df['pm25'].rolling(window=10).mean()
df['pm10_rolling_mean_5'] = df['pm10'].rolling(window=5).mean()
df['pm10_rolling_mean_10'] = df['pm10'].rolling(window=10).mean()
df['temp_rolling_mean_5'] = df['temperature'].rolling(window=5).mean()
df['humidity_rolling_mean_5'] = df['humidity'].rolling(window=5).mean()

# Rate of change
df['pm25_rate'] = df['pm25'] - df['pm25_lag1']
df['co_rate'] = df['co'] - df['co_lag1']
df['pm25_acceleration'] = df['pm25_rate'] - (df['pm25_lag1'] - df['pm25_lag2'])

# Interaction features
df['temp_humidity'] = df['temperature'] * df['humidity'] / 100
df['temp_pm25_interaction'] = df['temperature'] * df['pm25_lag1']
df['humidity_pm25_interaction'] = df['humidity'] * df['pm25_lag1']

# Cyclical time features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Drop nulls
print(f"\nBefore dropping nulls: {len(df):,}")
df_clean = df.dropna()
print(f"After dropping nulls: {len(df_clean):,}")

# Define features and targets
feature_columns = [
    'temperature', 'humidity', 'temp_lag1', 'humidity_lag1',
    'temp_rolling_mean_5', 'humidity_rolling_mean_5',
    'hour_of_day', 'day_of_week', 'is_weekend', 'time_of_day',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'pm25_lag1', 'pm25_lag2', 'pm25_lag3', 'pm25_lag5', 'pm25_lag10',
    'pm25_rolling_mean_5', 'pm25_rolling_mean_10',
    'pm25_rate', 'pm25_acceleration',
    'pm10_lag1', 'pm10_lag2', 'pm10_lag3',
    'pm10_rolling_mean_5', 'pm10_rolling_mean_10',
    'co_lag1', 'co_lag2', 'co_lag3',
    'lpg_lag1', 'lpg_lag2',
    'hydrogen_lag1',
    'co_rate',
    'temp_humidity', 'temp_pm25_interaction', 'humidity_pm25_interaction'
]

target_columns = ['pm25', 'pm10', 'co', 'lpg', 'hydrogen']

# Verify features exist
missing_features = [col for col in feature_columns if col not in df_clean.columns]
if missing_features:
    print(f"\nWarning: Missing features: {missing_features}")
    feature_columns = [col for col in feature_columns if col in df_clean.columns]

X = df_clean[feature_columns]
y = df_clean[target_columns]

print(f"\nFeature matrix: {X.shape}")
print(f"Target matrix: {y.shape}")

# Train/test split
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

# Train Random Forest model
print("\n" + "="*60)
print("Training Multi-Output Random Forest")
print("="*60)

base_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(base_rf)
model.fit(X_train, y_train)

print("Model training complete!")

# Make predictions
print("\nMaking predictions...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate model
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
        'model': 'Random Forest',
        'train_mae': float(train_mae),
        'train_rmse': float(train_rmse),
        'train_r2': float(train_r2),
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2)
    }

    print(f"\nTarget {target.upper()}:")
    print(f"   Training: MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}")
    print(f"   Testing:  MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}")

# Save model and results
print("\nSaving model and results...")
joblib.dump(model, 'random_forest_model.pkl')
print("Saved: random_forest_model.pkl")

with open('random_forest_features.json', 'w') as f:
    json.dump(feature_columns, f, indent=2)

with open('random_forest_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save full results
predictions_summary = {
    'model_type': 'Random Forest Regressor',
    'targets': target_columns,
    'features_count': len(feature_columns),
    'training_samples': len(X_train),
    'testing_samples': len(X_test),
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt'
    },
    'metrics': results,
    'timestamp': datetime.now().isoformat()
}

with open('random_forest_results.json', 'w') as f:
    json.dump(predictions_summary, f, indent=2)
print("Saved: random_forest_results.json")

# Generate visualizations
print("\nGenerating Random Forest visualizations...")

# 1. Actual vs Predicted
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
fig.suptitle('Random Forest Model: Actual vs Predicted Values', fontsize=16, fontweight='bold')

for i, target in enumerate(target_columns):
    sample_size = min(500, len(y_test))
    axes[i].scatter(y_test.iloc[:sample_size, i], y_pred_test[:sample_size, i],
                    alpha=0.3, s=10, color='#48bb78')
    min_val = min(y_test.iloc[:, i].min(), y_pred_test[:, i].min())
    max_val = max(y_test.iloc[:, i].max(), y_pred_test[:, i].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[i].set_xlabel(f'Actual {target.upper()}')
    axes[i].set_ylabel(f'Predicted {target.upper()}')
    axes[i].set_title(f'{target.upper()} - R2={results[target]["test_r2"]:.3f}', fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[5].axis('off')
plt.tight_layout()
plt.savefig('random_forest_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print("Saved: random_forest_actual_vs_predicted.png")
plt.close()

# 2. Time series comparison
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle('Random Forest Model: Time Series Predictions vs Actual', fontsize=16, fontweight='bold')

for i, target in enumerate(target_columns):
    sample_size = min(300, len(y_test))
    axes[i].plot(y_test.iloc[:sample_size, i].values, label='Actual',
                 color='#4299e1', linewidth=1, alpha=0.8)
    axes[i].plot(y_pred_test[:sample_size, i], label='Random Forest Prediction',
                 color='#e53e3e', linewidth=1, linestyle='--', alpha=0.8)
    axes[i].set_title(f'{target.upper()} - Test Set Predictions')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Value')
    axes[i].legend(loc='best')
    axes[i].grid(True, alpha=0.3)

axes[5].axis('off')
plt.tight_layout()
plt.savefig('random_forest_timeseries_predictions.png', dpi=150, bbox_inches='tight')
print("Saved: random_forest_timeseries_predictions.png")
plt.close()

# 3. Error distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
fig.suptitle('Random Forest Model: Prediction Error Distributions', fontsize=16, fontweight='bold')

for i, target in enumerate(target_columns):
    errors = y_test.iloc[:, i].values - y_pred_test[:, i]
    axes[i].hist(errors, bins=50, color='#48bb78', alpha=0.7, edgecolor='black')
    axes[i].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[i].axvline(errors.mean(), color='orange', linestyle='-', linewidth=2,
                    label=f'Mean Error: {errors.mean():.4f}')
    axes[i].set_title(f'{target.upper()} - Error Distribution', fontweight='bold')
    axes[i].set_xlabel('Prediction Error')
    axes[i].set_ylabel('Frequency')
    axes[i].text(0.95, 0.95, f'MAE: {results[target]["test_mae"]:.4f}\nR2: {results[target]["test_r2"]:.4f}',
                 transform=axes[i].transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    axes[i].legend()

axes[5].axis('off')
plt.tight_layout()
plt.savefig('random_forest_error_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: random_forest_error_distributions.png")
plt.close()

# 4. Performance bar chart
fig, ax = plt.subplots(figsize=(12, 6))
targets = [t.upper() for t in target_columns]
mae_values = [results[t]['test_mae'] for t in target_columns]
rmse_values = [results[t]['test_rmse'] for t in target_columns]
r2_values = [results[t]['test_r2'] for t in target_columns]

x = np.arange(len(targets))
width = 0.25

bars1 = ax.bar(x - width, mae_values, width, label='MAE', color='#48bb78')
bars2 = ax.bar(x, rmse_values, width, label='RMSE', color='#e53e3e')
bars3 = ax.bar(x + width, r2_values, width, label='R2', color='#4299e1')

ax.set_xlabel('Target Variable', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Random Forest Model: Performance by Target', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend()
ax.grid(True, alpha=0.3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('random_forest_performance_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: random_forest_performance_comparison.png")
plt.close()

# 5. Feature importance
print("\nCalculating feature importance...")
pm25_model = model.estimators_[0]
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': pm25_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values, color='#48bb78')
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance', fontsize=12)
plt.title('Random Forest Model: Top 15 Feature Importances for PM2.5 Prediction',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png', dpi=150, bbox_inches='tight')
print("Saved: random_forest_feature_importance.png")
plt.close()

feature_importance_dict = feature_importance.to_dict('records')
with open('random_forest_feature_importance.json', 'w') as f:
    json.dump(feature_importance_dict, f, indent=2)

# Update model comparison
import os
comparison_data = {}
if os.path.exists('model_comparison_all.json'):
    with open('model_comparison_all.json', 'r') as f:
        comparison_data = json.load(f)

comparison_data['Random Forest'] = results

with open('model_comparison_all.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)
print("Saved: model_comparison_all.json")

# Final summary
print("\n" + "="*60)
print("RANDOM FOREST MODEL COMPLETE!")
print("="*60)

print("\nMODEL PERFORMANCE SUMMARY:")
print("-" * 70)
print(f"{'Target':<12} {'MAE':<12} {'RMSE':<12} {'R2':<12} {'Status':<12}")
print("-" * 70)
for target in target_columns:
    r2 = results[target]['test_r2']
    if r2 >= 0.8:
        status = "Excellent"
    elif r2 >= 0.6:
        status = "Good"
    elif r2 >= 0.4:
        status = "Fair"
    else:
        status = "Poor"
    print(f"{target.upper():<12} {results[target]['test_mae']:<12.4f} "
          f"{results[target]['test_rmse']:<12.4f} {results[target]['test_r2']:<12.4f} {status:<12}")
print("-" * 70)

print("\nFiles saved:")
print("   - random_forest_model.pkl")
print("   - random_forest_metrics.json")
print("   - random_forest_results.json")
print("   - random_forest_*.png (5 visualization files)")
print("   - model_comparison_all.json (updated)")

print("\n" + "="*60)
