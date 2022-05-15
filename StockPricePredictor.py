"""
Stock Price Prediction

This program predicts the future date price of a stock. You begin by entering a stocks ticker symbol, 
the stock has to be listed on the stock market after 2014, 1, 1. 

"""

import math
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def Main():
    plt.style.use('seaborn-white')
    start = dt.datetime(2014, 1, 1)  # First day close price
    end = dt.datetime(2022, 1, 1)  # Last day close price
    Compony = \
        input('Enter The Ticker Symbol of The Stock You Want to Predict '
              )  # Asks user for ticker symbol
    df = web.DataReader(Compony, 'yahoo', start, end)  # Reads stock data from yahoo
    plt.figure(figsize=(16, 8))  # Graph size
    plt.title(Compony)  # Title of graph
    plt.xlabel('Days')  # X-axis Title
    plt.ylabel('Close Price USD ($)')  # Y-axis title
    plt.plot(df['Close'])  # Only displays the close prices
    plt.show()  # Displays the graph
    data = df.filter(['Close'])  # Converts Dataframe to only one column
    print (data)  # Prints close data
    dataset = data.values  # Converts Dataframe to numpy array
    training_data_len = math.ceil(len(dataset) * .8)  # Number of rows to train the LSTM model (80% of total data)
    scaler = MinMaxScaler(feature_range=(0, 1))  # Scaling the input data
    scaled_data = scaler.fit_transform(dataset)  # Holds dataset after being scaled, scaler.fit_transform finds the minimum and maximum for scaling based on the two values in feature_range=(0,1)
    train_data = scaled_data[0:training_data_len, :]  # Scaled training data set
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])  # Will contain 60 values from 0 to 59
        y_train.append(train_data[i, 0])
    (x_train, y_train) = (np.array(x_train), np.array(y_train))  # Converting the x_train and y_train to numpy arrays
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],
                         1))  # Reshaping data beacsue the LSTM model needs 3 deminsions
    global model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')  # optimizer is used to improve upon the loss function. Thee loss function measures how well it did during training
    model.fit(x_train, y_train, batch_size=1, epochs=1)  # Traing the model, epochs is the number of itterations
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]  # now contains the 61st value
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])  # now the x_test data set contains the past 60 values
    x_test = np.array(x_test)  # Converting the data to a numpy array so it will work in the LSTM model
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # Converting from 2d to 3d so it will work in the LSTM model
    predictions = model.predict(x_test)  # Getting the models predicted price value
    predictions = scaler.inverse_transform(predictions)  # This value should be the same as the test_y data set
    RSME = np.sqrt(np.mean(predictions - y_test) ** 2)  # Evaluate the model using RSME (root mean squared error 0 would mean a perfect prediction)
    print ('The Root Mean Squared Error Value is')
    print (RSME)
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    plt.figure(figsize=(16, 8))
    plt.title(Compony + ' Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD$')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training', 'Actual Price', 'Prediction'],
               loc='upper left')
    plt.show()
    StockQuote = web.DataReader(Compony, data_source='yahoo',
                                start='2014, 1, 1', end='2022, 1, 16')
    NewDf = StockQuote.filter(['Close'])  # Creating a new dataframe
    Last60 = NewDf[-60:].values  # Gathering the last 60 day closing prices, and converting them to an array
    Last_60_Days_Scaled = scaler.transform(Last60)  # Scaling the data between 0 and 1
    X_test = []
    X_test.append(Last_60_Days_Scaled)  # Append the last 60 days
    X_test = np.array(X_test)  # Converting the X_test data set to a numpy array, so it will work in the LSTM model
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshaping the data to 3d so it will work in the LSTM model
    PredictedPrice = model.predict(X_test)
    PredictedPrice = scaler.inverse_transform(PredictedPrice)  # Removing the scaling
    print ('\n')
    print ('This is the Predicted Price:')
    print (PredictedPrice)  # This is the predicted price on 2022, 1, 16
    print ('\n')
    StockQuoteTwo = web.DataReader(Compony, data_source='yahoo',
                                   start='2022, 1, 15',
                                   end='2022, 1, 15')  # The date here will only work when the program predicts a past date, since this part of the code lists the actual closing price, it will result in a error if future date is added here. When you want to predict the future date, put the date in the past and ignore the 'Actual Price' that prints. The predicted price wont be affected
    Final_Stock_quote2 = StockQuoteTwo['Close']
    print ('\n')
    print ('This is the Actual Price:')
    print (Final_Stock_quote2)


Main()

