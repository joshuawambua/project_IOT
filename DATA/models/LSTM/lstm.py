import numpy as np
import pandas as pd
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =======================
# 1. CONNECT TO MYSQL
# =======================
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Root@user123",
    database="aq_highres"
)

query = """
SELECT temperature, humidity, pm25, pm10, mq7_adc, mq2_adc, mq8_adc
FROM air_quality_data
ORDER BY id ASC
"""

df = pd.read_sql(query, conn)
conn.close()

print("Data loaded:", df.shape)

# =======================
# 2. CLEAN DATA
# =======================
df = df.dropna()

# =======================
# 3. SCALE DATA (VERY IMPORTANT FOR LSTM)
# =======================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# =======================
# 4. CREATE SEQUENCES
# =======================
def create_sequences(data, seq_length=20):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LEN = 20
X, y = create_sequences(data_scaled, SEQ_LEN)

# =======================
# 5. TRAIN / TEST SPLIT (80/20)
# =======================
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Train:", X_train.shape, "Test:", X_test.shape)

# =======================
# 6. BUILD LSTM MODEL
# =======================
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(X.shape[2]))  # output all features

model.compile(optimizer='adam', loss='mse')

# =======================
# 7. TRAIN MODEL
# =======================
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# =======================
# 8. PREDICTION
# =======================
y_pred = model.predict(X_test)

# invert scaling
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# =======================
# 9. EVALUATION METRICS
# =======================
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", mae)
print("RMSE:", rmse)

# =======================
# 10. FUTURE PREDICTION (NEXT STEP)
# =======================
last_sequence = data_scaled[-SEQ_LEN:]
last_sequence = np.expand_dims(last_sequence, axis=0)

future_prediction = model.predict(last_sequence)
future_prediction = scaler.inverse_transform(future_prediction)

print("\n===== NEXT STEP PREDICTION =====")
print("Temperature, Humidity, PM2.5, PM10, MQ7, MQ2, MQ8")
print(future_prediction[0])
