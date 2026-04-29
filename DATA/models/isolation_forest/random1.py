import os
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import firebase_admin
from firebase_admin import credentials, db

# =========================
# 1. ENV
# =========================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "YOUR_PASSWORD")
DB_NAME = os.getenv("DB_NAME", "aq_highres")

# =========================
# 2. MYSQL
# =========================
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME
)

query = """
SELECT temperature, humidity, pm25, pm10, mq2, mq7, mq8
FROM sensor_data
"""

df = pd.read_sql(query, conn)

print("Data loaded:", df.shape)

# =========================
# 3. CLEANING
# =========================
df = df.dropna()

df = df.clip(
    lower=df.quantile(0.01),
    upper=df.quantile(0.99),
    axis=1
)

# =========================
# 4. FEATURES / TARGET
# =========================
X = df.drop(columns=["pm25"])
y = df["pm25"]

# =========================
# 5. TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# =========================
# 6. RANDOM FOREST MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# 7. PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 8. METRICS
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== RANDOM FOREST =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# =========================
# 9. FUTURE PREDICTION
# =========================
latest = X.iloc[-1:].values
future_pred = model.predict(latest)[0]

print("Predicted PM2.5:", future_pred)

# =========================
# 10. FIREBASE
# =========================
cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://YOUR_PROJECT.firebaseio.com/"
})

db.reference("/predictions_random_forest").set({
    "pm25_predicted": float(future_pred),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "model": "random_forest"
})

print("✅ Random Forest pushed to Firebase")
