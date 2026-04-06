import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
data = df['Value'].values.reshape(-1, 1)

# Split data
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ARIMA Model
print("Training ARIMA Model...")
arima_model = ARIMA(train_data, order=(5, 1, 2))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test_data))

# LSTM Model
print("Training LSTM Model...")
X_train, y_train = [], []
for i in range(60, len(scaled_data[:train_size])):
    X_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    LSTM(50),
    Dense(25),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Predictions
X_test, y_test = [], []
for i in range(60, len(scaled_data)):
    X_test.append(scaled_data[i-60:i, 0])
    y_test.append(scaled_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

lstm_predictions = lstm_model.predict(X_test[-len(test_data):])
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Evaluation
arima_mse = mean_squared_error(test_data, arima_forecast)
lstm_mse = mean_squared_error(test_data[-len(lstm_predictions):], lstm_predictions)
print(f"ARIMA MSE: {arima_mse:.4f}")
print(f"LSTM MSE: {lstm_mse:.4f}")

# Visualization
plt.figure(figsize=(14, 5))
plt.plot(df['Date'][-len(test_data):], test_data, label='Actual', linewidth=2)
plt.plot(df['Date'][-len(test_data):], arima_forecast, label='ARIMA Forecast', linewidth=2)
plt.plot(df['Date'][-len(lstm_predictions):], lstm_predictions, label='LSTM Forecast', linewidth=2)
plt.legend()
plt.title('Time Series Forecasting: ARIMA vs LSTM')
plt.xlabel('Date')
plt.ylabel('Value')
plt.savefig('forecast_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Forecasting complete!")