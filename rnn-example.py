from math import sin

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from matplotlib import pyplot


def seq_func(x):
    return sin(x)


def fit_lstm(train_x, train_y, nb_epoch, neurons):
    x, y = np.array(train_x), np.array(train_y)
    x = x.reshape(len(x), 1, 1)
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(1, 1, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(x, y, epochs=1, batch_size=1, verbose=1, shuffle=False)
        model.reset_states()
    return model


series = [seq_func(x) for x in range(36)]
supervised = [seq_func(x + 1) for x in range(len(series))]

lstm_model = fit_lstm(series, supervised, 1000, 16)

train_reshaped = np.array(series).reshape(len(series), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

predictions = []
expectations = []
forecast_result = series[-1]
for i in range(36):
    forecast_result_reshaped = np.array([forecast_result]).reshape(1, 1, 1)
    forecast_result = lstm_model.predict(forecast_result_reshaped)[0, 0]
    predictions.append(forecast_result)
    expected = seq_func(len(series) + i)
    expectations.append(expected)
    print('x=%d, Predicted=%f, Expected=%f' % (len(series) + i, forecast_result, expected))

pyplot.plot(series + expectations)
# pyplot.plot(series + predictions)
pyplot.show()
