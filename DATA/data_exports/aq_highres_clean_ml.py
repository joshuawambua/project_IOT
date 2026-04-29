import mysql.connector
import pandas as pd
import numpy as np

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Root@user123",
    database="aq_highres"
)

query = """
SELECT *
FROM high_res_readings
ORDER BY reading_time
"""

df = pd.read_sql(query, conn)
conn.close()

df = df.sort_values("reading_time")

mq_cols = ["mq2_adc", "mq7_adc", "mq8_adc"]
env_cols = ["temperature", "humidity", "pm25", "pm10"]

df[mq_cols + env_cols] = df[mq_cols + env_cols].replace(0, np.nan)

df.loc[df["temperature"] < -10, "temperature"] = np.nan
df.loc[df["humidity"] <= 0, "humidity"] = np.nan
df.loc[df["humidity"] > 100, "humidity"] = np.nan

df = df.ffill().bfill()

df["mq2_adc"] = df["mq2_adc"].clip(0, 1500)
df["mq7_adc"] = df["mq7_adc"].clip(0, 1500)
df["mq8_adc"] = df["mq8_adc"].clip(0, 1500)

df["pm25"] = df["pm25"].clip(0, 500)
df["pm10"] = df["pm10"].clip(0, 600)

df["pm25_smooth"] = df["pm25"].rolling(3).mean()
df["pm10_smooth"] = df["pm10"].rolling(3).mean()

df.to_csv("aq_highres_clean_ml.csv", index=False)

print("Saved aq_highres_clean_ml.csv")
