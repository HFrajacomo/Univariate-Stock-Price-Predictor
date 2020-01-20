import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Cocacola_prepared.csv")
X = data[data.columns[:-1]]
y = data["target"]

scaler_x = MinMaxScaler(feature_range=(0,1))
scaler_y = MinMaxScaler(feature_range=(0,1))
X = scaler_x.fit_transform(X)
y = y.reshape(-1, 1)
y = scaler_y.fit_transform(y)


X_train = X[10:,:]
X_test = X[:10,:]
y_train = y[10:,:]
y_test = y[:10,:]

# Reshaping is THE MOST IMPORTANT PART CMON
X_train = np.reshape(X_train, (96, 1, 20))
X_test = np.reshape(X_test, (10,1,20))

cls = tf.keras.Sequential()
# SET INPUT_SHAPE TO THE RESHAPE FORMAT
cls.add(tf.keras.layers.LSTM(units=20, activation='tanh', input_shape=(1, 20), return_sequences=True))
cls.add(tf.keras.layers.Dropout(0.2))
cls.add(tf.keras.layers.LSTM(units=20, activation='tanh', input_shape=(1, 20), return_sequences=False))
cls.add(tf.keras.layers.Dropout(0.2))
cls.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cls.compile(optimzer='rmsprop', loss='mse')


cls.fit(X_train, y_train, batch_size=32, epochs=500, validation_split=0.05)
print(cls.evaluate(X_test, y_test))

predicted = []
truth = []

for entry in reversed(X_test):
	a = np.reshape(entry, (1,1,20))
	predicted.append(cls.predict(a)[0].tolist())

for entry in reversed(y_test):
	truth.append(entry.tolist())

predicted = scaler_y.inverse_transform(predicted)
truth = scaler_y.inverse_transform(truth)

plt.plot(range(len(predicted)), predicted, "r-", label="Predicted")
plt.plot(range(len(predicted)), truth, "b-", label="Truth")
plt.ylabel("Predicted Values")
plt.xlabel("Time")
plt.legend()
plt.show()
