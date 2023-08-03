import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime as dt

def fetch_stock_data(stock_symbol):
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=365 * 5)  # Data for the past 5 years
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

def perform_stock_prediction(stock_symbol):
    try:
        stock_data = fetch_stock_data(stock_symbol)
        output_var = pd.DataFrame(stock_data['Adj Close'])
        # Selecting the Features
        features = ['Open', 'High', 'Low', 'Volume']
        scaler = MinMaxScaler()
        feature_transform = scaler.fit_transform(stock_data[features])
        feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=stock_data.index)
        # Splitting to Training set and Test set
        timesplit = TimeSeriesSplit(n_splits=10)
        for train_index, test_index in timesplit.split(feature_transform):
            X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index) + len(test_index))]
            y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index) + len(test_index))].values.ravel()

        # Process the data for LSTM
        trainX = np.array(X_train)
        testX = np.array(X_test)
        X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

        lstm = Sequential()
        lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
        lstm.add(Dense(1))
        lstm.compile(loss='mean_squared_error', optimizer='adam')
        history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

        # Extend the test data to include the data for the next year
        next_year_start_date = stock_data.index[-1] + dt.timedelta(days=1)
        next_year_end_date = next_year_start_date + dt.timedelta(days=365)
        next_year_data = yf.download(stock_symbol, start=next_year_start_date, end=next_year_end_date)
        next_year_feature_transform = scaler.transform(next_year_data[features])
        next_year_feature_transform = pd.DataFrame(columns=features, data=next_year_feature_transform, index=next_year_data.index)
        X_next_year = np.array(next_year_feature_transform)
        X_next_year = X_next_year.reshape(X_next_year.shape[0], 1, X_next_year.shape[1])

        # LSTM Prediction for the next year
        y_next_year_pred = lstm.predict(X_next_year)

        return y_next_year_pred, None
    except Exception as e:
        return None, str(e)
