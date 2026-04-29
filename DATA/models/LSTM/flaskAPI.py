from flask import Flask, jsonify
import numpy as np
import mysql.connector
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = Flask(__name__)

# =====================
# LOAD DATA
# =====================
def load_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Root@user123",
        database="aq_highres"
    )

    df = pd.read_sql("""
        SELECT temperature, humidity, pm25, pm10, mq7_adc, mq2_adc, mq8_adc
        FROM air_quality_data
        ORDER BY id ASC
    """, conn)

    conn.close()
    return df.dropna()

# =====================
# SIMPLE LSTM LOAD (already trained model)
# =====================
model = tf.keras.models.load_model("lstm_model.h5")

scaler = MinMaxScaler()

SEQ_LEN = 20

def prepare_input(df):
    data = scaler.fit_transform(df)
    seq = data[-SEQ_LEN:]
    return np.expand_dims(seq, axis=0)

@app.route("/predict", methods=["GET"])
def predict():
    df = load_data()
    input_data = prepare_input(df)

    pred = model.predict(input_data)
    pred = scaler.inverse_transform(pred)

    return jsonify({
        "temperature": float(pred[0][0]),
        "humidity": float(pred[0][1]),
        "pm25": float(pred[0][2]),
        "pm10": float(pred[0][3]),
        "mq7": float(pred[0][4]),
        "mq2": float(pred[0][5]),
        "mq8": float(pred[0][6])
    })

if __name__ == "__main__":
    app.run(debug=True)
