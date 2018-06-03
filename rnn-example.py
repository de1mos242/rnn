from math import sin

import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat


def seq_func(x):
    return sin(x)


def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def fit_lstm(train, nb_epoch, neurons):
    x, y = train[:, 0:-1], train[:, -1]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(1, 1, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(x, y, epochs=1, batch_size=1, verbose=1, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, x):
    x = np.array([x]).reshape(1, 1, 1)
    y = model.predict(x)
    return y[0, 0]


series = DataFrame([seq_func(x) for x in range(48)])

supervised = timeseries_to_supervised(series.values)
supervised_values = supervised.values

# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# fit the model
lstm_model = fit_lstm(train, 1000, 16)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train[:, 0].reshape(len(train), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
expectations = list()
forecast_result = train[-1, 1]
for i in range(36):
    forecast_result = forecast_lstm(lstm_model, forecast_result)
    predictions.append(forecast_result)
    expected = seq_func(len(train) + i)
    expectations.append(expected)
    print('x=%d, Predicted=%f, Expected=%f' % (len(train) + i, forecast_result, expected))

a = [x for x in raw_values[:-12]]
pyplot.plot(a + [x for x in expectations])
pyplot.plot(a + [x for x in predictions])
pyplot.show()
