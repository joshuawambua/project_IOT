import os
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

from sklearn.ensemble import IsolationForest

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

# =========================
# 4. MODEL INPUT
# =========================
X = df.values

# =========================
# 5. ISOLATION FOREST
# =========================
model = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # 5% anomalies expected
    random_state=42
)

model.fit(X)

# =========================
# 6. PREDICT ANOMALY
# =========================
anomaly_scores = model.predict(X)

# -1 = anomaly, 1 = normal
df["anomaly"] = anomaly_scores

anomaly_count = np.sum(anomaly_scores == -1)

print("\n===== ISOLATION FOREST =====")
print("Anomalies detected:", anomaly_count)

# =========================
# 7. LATEST STATUS
# =========================
latest = X[-1:].reshape(1, -1)
latest_status = model.predict(latest)[0]

status = "ANOMALY" if latest_status == -1 else "NORMAL"

print("Latest status:", status)

# =========================
# 8. FIREBASE PUSH
# =========================
cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://YOUR_PROJECT.firebaseio.com/"
})

db.reference("/anomaly_detection").set({
    "status": status,
    "anomalies_detected": int(anomaly_count)
})

print("✅ Anomaly results pushed to Firebase")
