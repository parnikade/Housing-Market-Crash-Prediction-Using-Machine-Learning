import random
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
random.seed(0)
np.random.seed(0)


def main():

    df = pd.read_csv('Price and rate.csv', parse_dates=True)
    df.drop(df.columns[0], axis=1, inplace=True)
    # print(df)
    # print(df.columns)
    house_price = df['price'].values

    house_price = house_price.reshape((-1, 1))
    # print(house_price)
    split = 0.85
    train_test_split = int(split*len(house_price))

    house_price_train = house_price[:train_test_split]
    house_price_test = house_price[train_test_split:]

    date_train = df['Date'][:train_test_split]
    date_test = df['Date'][train_test_split:]

    look_back = 5

    train_generator = TimeseriesGenerator(house_price_train, house_price_train,
                                          length=look_back, batch_size=20)
    test_generator = TimeseriesGenerator(house_price_test, house_price_test,
                                         length=look_back, batch_size=2)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, 1),
                   init='uniform'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    num_epochs = 250
    model.fit_generator(train_generator, epochs=num_epochs, verbose=2,
                        shuffle=False)

    l_train = np.asarray(range(len(date_train)))
    l_test = np.asarray(range(len(date_train),
                              len(date_test) + len(date_train)))

    prediction = model.predict_generator(test_generator)
    house_price_train = house_price_train.reshape((-1))
    house_price_test = house_price_test.reshape((-1))
    prediction = prediction.reshape((-1))
    print(len(date_test))
    # print()
    print(prediction)

    plt.plot(l_train, house_price_train,'r-')
    plt.plot(l_test, house_price_test, 'r--')
    plt.plot(l_test[2:len(prediction)+2], prediction, 'b-')


    plt.xlabel('Days starting from Jan 1990')
    plt.ylabel('House prices training vs testing vs predicted')
    plt.title('Days vs Prices')
    plt.show()

    # extended days prediction

    num_prediction = 12
    prediction_list = house_price[-look_back:]
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back - 1:]
    print('Prediction list')
    print(prediction_list)

    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date,
                                     periods=num_prediction + 1,
                                     freq=pd.offsets.MonthBegin(1),
                                     closed='right').tolist()

    prediction_dates = [t.strftime("%Y-%m-%d %H:%M:%S") + '.000000'
                        for t in prediction_dates]

    a_series = pd.Series(prediction_dates)
    date_test = date_test.append(a_series, ignore_index=True)

    prediction_dates_df = date_test[-num_prediction-1:]
    date_test = date_test[:-num_prediction]
    l_test = np.asarray(range(len(date_train),
                              len(date_test) + len(date_train)))
    l_prediction = np.asarray(range(len(l_test) + len(date_train) - 1,
                              len(date_train) + len(l_test)
                                    + len(prediction_dates_df) - 1))

    print(len(prediction_dates_df))
    print(len(prediction_list))

    plt.plot(l_train, house_price_train, 'b-')
    plt.plot(l_test, house_price_test, 'b-')
    plt.plot(l_prediction, prediction_list, 'g--')

    plt.xlabel('Days starting from Jan 1990')
    plt.ylabel('House prices predicted and actual')
    plt.title('Days vs Prices')

    plt.show()


main()

