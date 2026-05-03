#!/usr/bin/env python3
"""
Isolation Forest for Anomaly Detection in Air Quality Data
Identifies abnormal pollution patterns and sensor anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime

# Convert numpy types to Python types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

plt.style.use('seaborn-v0_8-darkgrid')

print("="*60)
print("ISOLATION FOREST - ANOMALY DETECTION")
print("="*60)

# Load cleaned data
print("\nLoading cleaned data...")
df = pd.read_csv('aq_highres_clean_ml.csv')
df['reading_time'] = pd.to_datetime(df['reading_time'])

print(f"Loaded {len(df):,} records")
print(f"Date range: {df['reading_time'].min()} to {df['reading_time'].max()}")

# Map sensor readings
df['co'] = df['mq7_voltage']
df['lpg'] = df['mq2_voltage']
df['hydrogen'] = df['mq8_voltage']

# Create features for anomaly detection
print("\nCreating features for anomaly detection...")

# Basic features
anomaly_features = [
    'pm25', 'pm10', 'co', 'lpg', 'hydrogen',
    'temperature', 'humidity'
]

# Add derived features
df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 0.001)
df['temp_humidity_product'] = df['temperature'] * df['humidity']
df['total_pollution'] = df['pm25'] + df['pm10'] + df['co'] + df['lpg'] + df['hydrogen']
df['pm25_change'] = df['pm25'].diff()
df['co_change'] = df['co'].diff()

# Add time features
df['hour'] = df['hour']
df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 20)).astype(int)

# Extended feature set for anomaly detection
anomaly_features_extended = anomaly_features + [
    'pm25_pm10_ratio', 'temp_humidity_product', 'total_pollution',
    'pm25_change', 'co_change', 'is_night'
]

# Drop nulls
df_clean = df.dropna()
print(f"\nData after dropping nulls: {len(df_clean):,}")

# Prepare data for Isolation Forest
X_anomaly = df_clean[anomaly_features_extended]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anomaly)

print(f"\nFeature matrix shape: {X_scaled.shape}")
print(f"Features used: {len(anomaly_features_extended)}")

# Train Isolation Forest
print("\n" + "="*60)
print("Training Isolation Forest for Anomaly Detection")
print("="*60)

isolation_forest = IsolationForest(
    contamination=0.05,  # Expect 5% anomalies
    random_state=42,
    n_estimators=100,
    max_samples='auto',
    bootstrap=False
)

# Train the model
isolation_forest.fit(X_scaled)
print("Model training complete!")

# Predict anomalies
print("\nDetecting anomalies...")
predictions = isolation_forest.predict(X_scaled)
anomaly_scores = isolation_forest.score_samples(X_scaled)

# Add results to dataframe
df_clean['anomaly'] = predictions
df_clean['anomaly_score'] = anomaly_scores
df_clean['is_anomaly'] = (predictions == -1).astype(int)

# Calculate statistics
n_anomalies = int((predictions == -1).sum())
n_normal = int((predictions == 1).sum())
anomaly_percentage = float((n_anomalies / len(df_clean)) * 100)

print(f"\nAnomaly Detection Results:")
print(f"   Normal samples: {n_normal:,} ({100-anomaly_percentage:.1f}%)")
print(f"   Anomalies detected: {n_anomalies:,} ({anomaly_percentage:.1f}%)")

# Analyze anomalies by pollutant levels
print("\nAnomaly Characteristics:")
anomaly_stats = {}
for pollutant in ['pm25', 'pm10', 'co', 'lpg', 'hydrogen']:
    normal_mean = float(df_clean[df_clean['is_anomaly'] == 0][pollutant].mean())
    anomaly_mean = float(df_clean[df_clean['is_anomaly'] == 1][pollutant].mean())
    ratio = float(anomaly_mean / (normal_mean + 0.001))
    anomaly_stats[pollutant] = {
        'normal_mean': normal_mean,
        'anomaly_mean': anomaly_mean,
        'ratio': ratio
    }
    print(f"   {pollutant.upper()}: Anomalies are {ratio:.2f}x higher")

# Save model and results
print("\nSaving model and results...")
joblib.dump(isolation_forest, 'isolation_forest_model.pkl')
joblib.dump(scaler, 'isolation_forest_scaler.pkl')
print("Saved: isolation_forest_model.pkl, isolation_forest_scaler.pkl")

# Save results with proper serialization
results = {
    'model_type': 'Isolation Forest',
    'contamination': 0.05,
    'n_estimators': 100,
    'total_samples': int(len(df_clean)),
    'normal_samples': n_normal,
    'anomaly_samples': n_anomalies,
    'anomaly_percentage': anomaly_percentage,
    'features_used': anomaly_features_extended,
    'anomaly_statistics': anomaly_stats,
    'timestamp': datetime.now().isoformat()
}

with open('isolation_forest_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=convert_to_serializable)
print("Saved: isolation_forest_results.json")

# Save anomaly timestamps for review
anomaly_records = df_clean[df_clean['is_anomaly'] == 1][
    ['reading_time', 'pm25', 'pm10', 'co', 'lpg', 'hydrogen', 'temperature', 'humidity', 'anomaly_score']
].sort_values('anomaly_score')
anomaly_records.to_csv('isolation_forest_anomalies.csv', index=False)
print("Saved: isolation_forest_anomalies.csv")

# Generate visualizations
print("\nGenerating Isolation Forest visualizations...")

# 1. Anomaly timeline
fig, ax = plt.subplots(figsize=(15, 6))
normal_points = df_clean[df_clean['is_anomaly'] == 0]
anomaly_points = df_clean[df_clean['is_anomaly'] == 1]

ax.scatter(normal_points['reading_time'], normal_points['pm25'],
           c='blue', alpha=0.3, s=10, label='Normal')
ax.scatter(anomaly_points['reading_time'], anomaly_points['pm25'],
           c='red', alpha=0.7, s=50, label='Anomaly', marker='x')
ax.set_xlabel('Time')
ax.set_ylabel('PM2.5')
ax.set_title('Isolation Forest: PM2.5 Anomaly Detection Timeline', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('isolation_forest_anomaly_timeline.png', dpi=150, bbox_inches='tight')
print("Saved: isolation_forest_anomaly_timeline.png")
plt.close()

# 2. Anomaly score distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of anomaly scores
axes[0].hist(df_clean[df_clean['is_anomaly'] == 0]['anomaly_score'],
             bins=50, alpha=0.5, color='blue', label='Normal', density=True)
axes[0].hist(df_clean[df_clean['is_anomaly'] == 1]['anomaly_score'],
             bins=20, alpha=0.5, color='red', label='Anomaly', density=True)
axes[0].set_xlabel('Anomaly Score')
axes[0].set_ylabel('Density')
axes[0].set_title('Anomaly Score Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot of anomaly scores
anomaly_data = [df_clean[df_clean['is_anomaly'] == 0]['anomaly_score'].values,
                df_clean[df_clean['is_anomaly'] == 1]['anomaly_score'].values]
bp = axes[1].boxplot(anomaly_data, labels=['Normal', 'Anomaly'], patch_artist=True)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][1].set_facecolor('red')
axes[1].set_ylabel('Anomaly Score')
axes[1].set_title('Anomaly Score Comparison')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('isolation_forest_score_distribution.png', dpi=150, bbox_inches='tight')
print("Saved: isolation_forest_score_distribution.png")
plt.close()

# 3. Feature comparison (normal vs anomaly)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

pollutants = ['pm25', 'pm10', 'co', 'lpg', 'hydrogen', 'temperature']
for i, pollutant in enumerate(pollutants):
    normal_vals = df_clean[df_clean['is_anomaly'] == 0][pollutant].values
    anomaly_vals = df_clean[df_clean['is_anomaly'] == 1][pollutant].values

    axes[i].boxplot([normal_vals, anomaly_vals], labels=['Normal', 'Anomaly'], patch_artist=True)
    axes[i].set_ylabel(pollutant.upper())
    axes[i].set_title(f'{pollutant.upper()} Distribution')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Isolation Forest: Feature Comparison (Normal vs Anomaly)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('isolation_forest_feature_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: isolation_forest_feature_comparison.png")
plt.close()

# 4. Anomaly heatmap by hour
fig, ax = plt.subplots(figsize=(12, 6))
hourly_anomalies = df_clean.groupby('hour')['is_anomaly'].mean() * 100
ax.bar(hourly_anomalies.index, hourly_anomalies.values, color='#e53e3e', alpha=0.7)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Anomaly Percentage (%)')
ax.set_title('Isolation Forest: Anomaly Distribution by Hour', fontweight='bold')
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('isolation_forest_hourly_anomalies.png', dpi=150, bbox_inches='tight')
print("Saved: isolation_forest_hourly_anomalies.png")
plt.close()

# Update model comparison
import os
comparison_data = {}
if os.path.exists('model_comparison_all.json'):
    with open('model_comparison_all.json', 'r') as f:
        comparison_data = json.load(f)

comparison_data['Isolation Forest'] = {
    'anomaly_detection': True,
    'anomaly_percentage': anomaly_percentage,
    'total_anomalies': n_anomalies,
    'anomaly_stats': anomaly_stats
}

with open('model_comparison_all.json', 'w') as f:
    json.dump(comparison_data, f, indent=2, default=convert_to_serializable)
print("Updated: model_comparison_all.json")

# Final summary
# print("\n" + "="*60)
print("ISOLATION FOREST ANOMALY DETECTION COMPLETE!")
print("="*60)

print(f"\nSUMMARY:")
print(f"   Total samples analyzed: {len(df_clean):,}")
print(f"   Anomalies detected: {n_anomalies:,} ({anomaly_percentage:.1f}%)")
print(f"\n   Anomaly characteristics:")
for pollutant, stats in anomaly_stats.items():
    print(f"   - {pollutant.upper()}: {stats['ratio']:.2f}x higher in anomalies")

print("\nFiles saved:")
print("   - isolation_forest_model.pkl")
print("   - isolation_forest_scaler.pkl")
print("   - isolation_forest_results.json")
print("   - isolation_forest_anomalies.csv")
print("   - isolation_forest_anomaly_timeline.png")
print("   - isolation_forest_score_distribution.png")
print("   - isolation_forest_feature_comparison.png")
print("   - isolation_forest_hourly_anomalies.png")

print("\n" + "="*60)
