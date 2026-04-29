#!/usr/bin/env python3
"""
LSTM Model for Multi-Target Prediction
Predicts: PM2.5, PM10, CO, LPG, Hydrogen using sequential data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import json
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')

print("="*60)
print("LSTM MODEL FOR AIR QUALITY PREDICTION")
print("="*60)

# Load cleaned data
print("\nLoading cleaned data...")
df = pd.read_csv('../decision_tree/aq_highres_clean_ml.csv')
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

X = df_clean[feature_columns].values
y = df_clean[target_columns].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target matrix shape: {y.shape}")

# Train/test split (chronological)
print("\n" + "="*60)
print("TRAIN/TEST SPLIT (Chronological)")
print("="*60)

split_idx = int(len(df_clean) * 0.8)
X_train_raw = X[:split_idx]
X_test_raw = X[split_idx:]
y_train_raw = y[:split_idx]
y_test_raw = y[split_idx:]

print(f"Training set: {len(X_train_raw):,} records ({len(X_train_raw)/len(df_clean)*100:.1f}%)")
print(f"Testing set: {len(X_test_raw):,} records ({len(X_test_raw)/len(df_clean)*100:.1f}%)")

# Scale the data
print("\nScaling features and targets...")
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_train = feature_scaler.fit_transform(X_train_raw)
X_test = feature_scaler.transform(X_test_raw)
y_train = target_scaler.fit_transform(y_train_raw)
y_test = target_scaler.transform(y_test_raw)

print("Features and targets scaled to range [0, 1]")

# Reshape for LSTM (samples, timesteps, features)
# Use a sliding window of 10 timesteps
sequence_length = 10

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

print(f"\nCreating sequences with length {sequence_length}...")
X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

print(f"Training sequences: {X_train_seq.shape}")
print(f"Testing sequences: {X_test_seq.shape}")

# Build LSTM model
print("\n" + "="*60)
print("Building LSTM Model")
print("="*60)

# Input layer
inputs = Input(shape=(sequence_length, X_train.shape[1]))

# Bidirectional LSTM layers
x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.2)(x)
x = Bidirectional(LSTM(32))(x)
x = Dropout(0.2)(x)

# Dense layers
x = Dense(64, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(32, activation='relu')(x)

# Output layer (5 targets)
outputs = Dense(5, activation='linear')(x)

# Create model
model = Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train model
print("\n" + "="*60)
print("Training LSTM Model")
print("="*60)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("Model training complete!")

# Make predictions
print("\nMaking predictions...")
y_pred_scaled = model.predict(X_test_seq)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_actual = target_scaler.inverse_transform(y_test_seq)

# Evaluate model
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

results = {}
for i, target in enumerate(target_columns):
    test_mae = mean_absolute_error(y_test_actual[:, i], y_pred[:, i])
    test_rmse = np.sqrt(mean_squared_error(y_test_actual[:, i], y_pred[:, i]))
    test_r2 = r2_score(y_test_actual[:, i], y_pred[:, i])

    results[target] = {
        'model': 'LSTM',
        'test_mae': float(test_mae),
        'test_rmse': float(test_rmse),
        'test_r2': float(test_r2)
    }

    print(f"\nTarget {target.upper()}:")
    print(f"   Testing: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}")

# Save model and results
print("\nSaving model and results...")
model.save('lstm_model.h5')
print("Saved: lstm_model.h5")

joblib.dump(feature_scaler, 'lstm_feature_scaler.pkl')
joblib.dump(target_scaler, 'lstm_target_scaler.pkl')
print("Saved: lstm_feature_scaler.pkl, lstm_target_scaler.pkl")

with open('lstm_features.json', 'w') as f:
    json.dump(feature_columns, f, indent=2)

with open('lstm_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save full results
predictions_summary = {
    'model_type': 'LSTM (Bidirectional)',
    'targets': target_columns,
    'features_count': len(feature_columns),
    'training_samples': len(X_train_seq),
    'testing_samples': len(X_test_seq),
    'sequence_length': sequence_length,
    'hyperparameters': {
        'lstm_units': [128, 64, 32],
        'dropout': [0.2, 0.2, 0.2],
        'bidirectional': True,
        'epochs': len(history.history['loss']),
        'batch_size': 32
    },
    'metrics': results,
    'timestamp': datetime.now().isoformat()
}

with open('lstm_results.json', 'w') as f:
    json.dump(predictions_summary, f, indent=2)
print("Saved: lstm_results.json")

# Generate visualizations
print("\nGenerating LSTM visualizations...")

# 1. Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LSTM Model: Training History', fontsize=16, fontweight='bold')

axes[0].plot(history.history['loss'], label='Training Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Loss Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Training MAE', color='blue')
axes[1].plot(history.history['val_mae'], label='Validation MAE', color='red')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('MAE Over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=150, bbox_inches='tight')
print("Saved: lstm_training_history.png")
plt.close()

# 2. Actual vs Predicted
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
fig.suptitle('LSTM Model: Actual vs Predicted Values', fontsize=16, fontweight='bold')

for i, target in enumerate(target_columns):
    sample_size = min(500, len(y_test_actual))
    axes[i].scatter(y_test_actual[:sample_size, i], y_pred[:sample_size, i],
                    alpha=0.3, s=10, color='#9b59b6')
    min_val = min(y_test_actual[:, i].min(), y_pred[:, i].min())
    max_val = max(y_test_actual[:, i].max(), y_pred[:, i].max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[i].set_xlabel(f'Actual {target.upper()}')
    axes[i].set_ylabel(f'Predicted {target.upper()}')
    axes[i].set_title(f'{target.upper()} - R2={results[target]["test_r2"]:.3f}', fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()

axes[5].axis('off')
plt.tight_layout()
plt.savefig('lstm_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
print("Saved: lstm_actual_vs_predicted.png")
plt.close()

# 3. Time series comparison
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()
fig.suptitle('LSTM Model: Time Series Predictions vs Actual', fontsize=16, fontweight='bold')

for i, target in enumerate(target_columns):
    sample_size = min(300, len(y_test_actual))
    axes[i].plot(y_test_actual[:sample_size, i], label='Actual',
                 color='#4299e1', linewidth=1, alpha=0.8)
    axes[i].plot(y_pred[:sample_size, i], label='LSTM Prediction',
                 color='#e53e3e', linewidth=1, linestyle='--', alpha=0.8)
    axes[i].set_title(f'{target.upper()} - Test Set Predictions')
    axes[i].set_xlabel('Sample Index')
    axes[i].set_ylabel('Value')
    axes[i].legend(loc='best')
    axes[i].grid(True, alpha=0.3)

axes[5].axis('off')
plt.tight_layout()
plt.savefig('lstm_timeseries_predictions.png', dpi=150, bbox_inches='tight')
print("Saved: lstm_timeseries_predictions.png")
plt.close()

# 4. Error distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
fig.suptitle('LSTM Model: Prediction Error Distributions', fontsize=16, fontweight='bold')

for i, target in enumerate(target_columns):
    errors = y_test_actual[:, i] - y_pred[:, i]
    axes[i].hist(errors, bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
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
plt.savefig('lstm_error_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: lstm_error_distributions.png")
plt.close()

# 5. Performance bar chart
fig, ax = plt.subplots(figsize=(12, 6))
targets = [t.upper() for t in target_columns]
mae_values = [results[t]['test_mae'] for t in target_columns]
rmse_values = [results[t]['test_rmse'] for t in target_columns]
r2_values = [results[t]['test_r2'] for t in target_columns]

x = np.arange(len(targets))
width = 0.25

bars1 = ax.bar(x - width, mae_values, width, label='MAE', color='#9b59b6')
bars2 = ax.bar(x, rmse_values, width, label='RMSE', color='#e53e3e')
bars3 = ax.bar(x + width, r2_values, width, label='R2', color='#4299e1')

ax.set_xlabel('Target Variable', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('LSTM Model: Performance by Target', fontsize=14, fontweight='bold')
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
plt.savefig('lstm_performance_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: lstm_performance_comparison.png")
plt.close()

# Update model comparison
import os
comparison_data = {}
if os.path.exists('../isolation_forest/model_comparison_all.json'):
    with open('../isolation_forest/model_comparison_all.json', 'r') as f:
        comparison_data = json.load(f)

comparison_data['LSTM'] = results

with open('../isolation_forest/model_comparison_all.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)
print("Updated: model_comparison_all.json")

# Final summary
print("\n" + "="*60)
print("LSTM MODEL COMPLETE!")
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
print("   - lstm_model.h5")
print("   - lstm_feature_scaler.pkl")
print("   - lstm_target_scaler.pkl")
print("   - lstm_features.json")
print("   - lstm_metrics.json")
print("   - lstm_results.json")
print("   - lstm_training_history.png")
print("   - lstm_actual_vs_predicted.png")
print("   - lstm_timeseries_predictions.png")
print("   - lstm_error_distributions.png")
print("   - lstm_performance_comparison.png")

print("\n" + "="*60)
