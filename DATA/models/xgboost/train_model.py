"""
XGBoost Model Training Script
Run this once to train and save your model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import json
import os

print("=" * 60)
print("XGBOOST MODEL TRAINING")
print("=" * 60)

# Check if data file exists
if not os.path.exists('aq_highres_clean_ml.csv'):
    print("\n❌ ERROR: aq_highres_clean_ml.csv not found!")
    print("   Please make sure your CSV file is in the current directory.")
    exit(1)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('aq_highres_clean_ml.csv')
print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

# Define features (matching your ESP32 sensors)
feature_cols = ['temperature', 'humidity', 'pm25', 'pm10', 'mq2_adc', 'mq7_adc', 'mq8_adc']

# Check if all features exist
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f"\n❌ Missing columns: {missing_cols}")
    print(f"\nAvailable columns: {df.columns.tolist()}")
    exit(1)

# Target variable
target_col = 'quality_flag'

if target_col not in df.columns:
    print(f"\n❌ Target column '{target_col}' not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

# Prepare data
print("\n🔧 Preparing data...")
X = df[feature_cols].copy()
y = df[target_col].copy()

# Remove rows with missing values
initial_rows = len(X)
X = X.dropna()
y = y.loc[X.index]

print(f"   Removed {initial_rows - len(X)} rows with missing values")
print(f"   Remaining: {len(X):,} rows")

# Check quality_flag values
unique_values = sorted(y.unique())
print(f"\n📊 Quality flag values found: {unique_values}")

# Encode labels (make them sequential: 0, 1, 2...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"   Encoded values: {sorted(np.unique(y_encoded))}")
print(f"   Mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Training: {len(X_train):,} samples")
print(f"   Test: {len(X_test):,} samples")

# Scale features
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\n🤖 Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)

model.fit(X_train_scaled, y_train)

# Evaluate
print("\n📊 Evaluating model...")
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)

print(f"\n{'='*50}")
print(f"RESULTS:")
print(f"{'='*50}")
print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"{'='*50}")

# Save model artifacts
print("\n💾 Saving model artifacts...")
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(scaler, 'xgboost_scaler.pkl')
joblib.dump(label_encoder, 'xgboost_label_encoder.pkl')

# Save results
results = {
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'features': feature_cols,
    'target': target_col,
    'original_classes': [int(c) for c in label_encoder.classes_],
    'encoded_classes': list(range(len(label_encoder.classes_))),
    'mapping': {int(k): int(v) for k, v in zip(label_encoder.classes_, range(len(label_encoder.classes_)))},
    'n_samples': len(X),
    'n_features': len(feature_cols),
    'n_classes': len(label_encoder.classes_)
}

with open('xgboost_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Saved files:")
print("   - xgboost_model.pkl")
print("   - xgboost_scaler.pkl")
print("   - xgboost_label_encoder.pkl")
print("   - xgboost_results.json")

print("\n🎉 Training complete! You can now run:")
print("   python control.py start")
