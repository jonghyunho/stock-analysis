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
num_features = 10
num_output = 2


def get_data(code, net):
    df = fdr.DataReader(code, '2016-01-01')
    df.columns = df.columns.str.lower()
    df = df.drop(columns=['change'])
    df = df[['close']]
    df.index.name = 'date'

    x = np.log(1 + df.pct_change()).copy()
    for period in [5, 10, 20, 60, 120]:
        x[f'ma{period}'] = df['close'] / \
            talib.SMA(df['close'], timeperiod=period) - 1.0

    x['rsi'] = get_RSI(df, 14) * 0.01
    x['rsi_signal'] = get_RSI(df, 9) * 0.01

    upper, _, low = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    x['bb_upper'] = df['close'] / upper - 1.0
    x['bb_low'] = df['close'] / low - 1.0
    x = x.dropna()

    x_values = []
    y_values = []
    for i in range(num_steps, len(x)):
        x_values.append([x.iloc[i-num_steps: i].values])

        y_val = x['ma20'][i]
        y_val = [1, 0] if y_val <= 0 else [0, 1]
        y_values.append(y_val)

    if net == 'dnn' or net == 'lstm':
        x_values = np.array(x_values).reshape(-1, num_steps, num_features)
    else:
        x_values = np.array(x_values).reshape(-1, num_steps, num_features, 1)
    y_values = np.array(y_values).reshape(-1, num_output)
    return x_values, y_values

net = 'cnn'

x_train, y_train = get_data('005930', net)
x_val, y_val = get_data('105560', net)
x_test, y_test = get_data('122630', net)

if net == 'dnn':
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
        tf.keras.layers.Dense(num_output, activation='sigmoid')
    ])
elif net == 'lstm':
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(num_steps, num_features), dropout=0.1,
                             return_sequences=True, stateful=False, activation='tanh'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(
            128, dropout=0.1, return_sequences=True, stateful=False, activation='tanh'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(
            128, dropout=0.1, return_sequences=True, stateful=False, activation='tanh'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(
            128, dropout=0.1, return_sequences=False, stateful=False, activation='tanh'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_output, activation='sigmoid')
    ])
else:  # cnn
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='sigmoid',
            input_shape=(num_steps, num_features, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding='same', activation='sigmoid'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            32, (3, 3), padding='same', activation='sigmoid'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            16, (3, 3), padding='same', activation='sigmoid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_output, activation='sigmoid')
    ])

model.compile(optimizer='adam', loss=tf.losses.binary_crossentropy,
              metrics=['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, batch_size=32,
                    epochs=1000, validation_data=(x_val, y_val))

model.save('./train/model.h5')

model.evaluate(x_test, y_test, verbose=2)

count = 0

num_total = 20

for i in range(num_total):
    if net == 'dnn' or net == 'lstm':
        x_test_value = x_test[i].reshape(-1, num_steps, num_features)
    else:
        x_test_value = x_test[i].reshape(-1, num_steps, num_features, 1)
    pred = model.predict(x_test_value)

    p = np.argmax(pred)
    y = np.argmax(y_test[i])

    if p == y:
        count += 1

    print(p, y)

print(f'count = {count} / {num_total}')
