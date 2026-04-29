import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Root@user123",
    database="aq_standard"
)

query = """
SELECT *
FROM standard_readings
ORDER BY reading_time
"""

df = pd.read_sql(query, conn)
conn.close()

df.to_csv("aq_standard_raw.csv", index=False)

print("Saved aq_standard_raw.csv")
