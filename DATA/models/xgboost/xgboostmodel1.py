import numpy as np
import pandas as pd
import mysql.connector
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import firebase_admin
from firebase_admin import credentials, db

# =========================
# 1. MYSQL CONNECTION
# =========================
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Root@user123",
    database="aq_highres"
)

query = """
SELECT temperature, humidity, pm25, pm10, mq7_adc, mq2_adc, mq8_adc
FROM high_res_readings
ORDER BY id ASC
"""

df = pd.read_sql(query, conn)
conn.close()

print("Data shape:", df.shape)

# =========================
# 2. CLEAN DATA
# =========================
df = df.dropna()

# Add LAG FEATURES (VERY IMPORTANT)
df['pm25_lag1'] = df['pm25'].shift(1)
df['pm25_lag2'] = df['pm25'].shift(2)

df = df.dropna()

# =========================
# 3. FEATURES & TARGET
# =========================
X = df[['temperature', 'humidity', 'mq7_adc', 'mq2_adc', 'mq8_adc',
        'pm25_lag1', 'pm25_lag2']]

y = df['pm25']

# =========================
# 4. TRAIN / TEST (80/20)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# =========================
# 5. MODEL
# =========================
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

# =========================
# 6. PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. METRICS
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== XGBOOST PERFORMANCE =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

# =========================
# 8. FUTURE PREDICTION
# =========================
latest_input = X.iloc[-1:].values
future_prediction = model.predict(latest_input)[0]

print("\nPredicted PM2.5:", future_prediction)

# =========================
# 9. SAVE LOCAL JSON
# =========================
results = {
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "prediction": float(future_prediction)
}

with open("xgb_results.json", "w") as f:
    json.dump(results, f)

print("Results saved to xgb_results.json")

# =========================
# 10. FIREBASE PUSH
# =========================
cred = credentials.Certificate(
    r"C:\Users\Joshua\Desktop\finals\DATA\models\xgboost\ecosphere-434df-firebase-adminsdk-fbsvc-02667124cf.json"
)

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://ecosphere-434df-default-rtdb.firebaseio.com/"
})

ref = db.reference("/ml/xgboost")

ref.set({
    "pm25_predicted": float(future_prediction),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "timestamp": int(pd.Timestamp.now().timestamp())
})

print("✅ Data pushed to Firebase successfully")
