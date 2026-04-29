import mysql.connector
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# =========================
# 1. CONNECT TO MYSQL
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

# =========================
# 3. FEATURES & TARGET
# =========================
X = df[['temperature', 'humidity', 'mq7_adc', 'mq2_adc', 'mq8_adc']]
y = df[['pm25', 'pm10']]

# =========================
# 4. TRAIN / TEST SPLIT (80/20)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. RANDOM FOREST MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=100,   # number of trees
    max_depth=12,       # control overfitting
    random_state=42,
    n_jobs=-1           # use all CPU cores
)

model.fit(X_train, y_train)

# =========================
# 6. PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. EVALUATION
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n===== RANDOM FOREST PERFORMANCE =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)
print("MAPE:", mape)

# =========================
# 8. SAVE RESULTS
# =========================
results = {
    "model": "Random Forest",
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2),
    "mape": float(mape),
    "actual_pm25": y_test['pm25'].tolist(),
    "pred_pm25": y_pred[:, 0].tolist()
}

with open("rf_results.json", "w") as f:
    json.dump(results, f)

print("Results saved to rf_results.json")

# =========================
# 9. PLOT: ACTUAL vs PREDICTED
# =========================
plt.figure()

plt.plot(y_test['pm25'].values[:100], label="Actual PM2.5")
plt.plot(y_pred[:100, 0], label="Predicted PM2.5")

plt.title("Random Forest: Actual vs Predicted PM2.5")
plt.xlabel("Samples")
plt.ylabel("PM2.5")
plt.legend()
plt.grid()

plt.show()

# =========================
# 10. METRICS BAR GRAPH
# =========================
metrics_names = ['MAE', 'RMSE', 'R²', 'MAPE']
metrics_values = [mae, rmse, r2, mape]

plt.figure()

plt.bar(metrics_names, metrics_values)

plt.title("Random Forest Evaluation Metrics")
plt.ylabel("Value")

plt.show()

# =========================
# 11. FEATURE IMPORTANCE
# =========================
importances = model.feature_importances_
features = X.columns

plt.figure()

plt.barh(features, importances)

plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")

plt.show()
