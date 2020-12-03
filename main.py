import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import FinanceDataReader as fdr

import tensorflow as tf
import talib


def get_RSI(df, period):
    close = df[['close']].copy()

    U = np.where(close.diff(1) > 0, close.diff(1), 0)
    D = np.where(close.diff(1) < 0, close.diff(1) * (-1), 0)

    AU = pd.DataFrame(U, index=close.index).rolling(period).mean()
    AD = pd.DataFrame(D, index=close.index).rolling(period).mean()
    rsi = AU.div(AD + AU) * 100

    return rsi


num_steps = 10
num_features = 8
num_output = 2


def get_data(code):
    df = fdr.DataReader(code, '2016-01-01')
    df.columns = df.columns.str.lower()
    df = df.drop(columns=['change'])
    df = df[['close']]
    df.index.name = 'date'

    x = np.log(1 + df.pct_change()).copy()
    for period in [5, 10, 20, 60, 120]:
        x[f'ma{period}'] = talib.SMA(
            df['close'], timeperiod=period) / df['close'] - 1.0

    x['rsi'] = get_RSI(df, 14) * 0.01
    x['rsi_signal'] = get_RSI(df, 9) * 0.01

    x = x.dropna()

    x_values = []
    y_values = []
    for i in range(num_steps, len(x)):
        x_values.append([x.iloc[i-num_steps: i].values])

        y_val = x['close'][i]
        y_val = [1, 0] if y_val <= 0 else [0, 1]
        y_values.append(y_val)

    x_values = np.array(x_values).reshape(-1, num_steps, num_features)
    y_values = np.array(y_values).reshape(-1, num_output)
    return x_values, y_values


x_train, y_train = get_data('005930')
x_val, y_val = get_data('105560')
x_test, y_test = get_data('122630')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(num_steps, num_features)),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(num_output, activation='linear')
])

model.compile(optimizer='adam', loss=tf.losses.mean_squared_error,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=32,
                    epochs=1000, validation_data=(x_val, y_val))

model.evaluate(x_test,  y_test, verbose=2)

count = 0

for i in range(20):
    pred = model.predict(x_test[i].reshape(-1, num_steps, num_features))

    p = np.argmax(pred)
    y = np.argmax(y_test[i])

    if p == y:
        count += 1

    print(p, y)

print(f'count = {count}')