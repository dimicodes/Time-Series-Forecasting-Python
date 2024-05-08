import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from lightgbm import LGBMRegressor


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.sort_index()  # Sorting the index to make sure it's in order

    # Resample to daily data, filling missing days with NaNs
    df = df.resample('D').mean()  # Change .mean() to another method if needed (e.g., .sum() or .first())

    # Optionally, fill NaN values if necessary
    df = df.ffill()  # Use .bfill() if you prefer backward filling

    return df


def scale_data(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train.reshape(-1, 1))
    test_sc = scaler.transform(test.reshape(-1, 1))
    return train_sc, test_sc, scaler


def build_nn_model(input_shape):
    nn_model = Sequential([
        Input(shape=(input_shape,)),  # Define the input shape directly
        Dense(12, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    nn_model.compile(loss='mean_squared_error', optimizer='adam')
    return nn_model


def build_lstm_model(input_shape):
    lstm_model = Sequential([
        Input(shape=(input_shape, 1)),  # Adjust the input shape for LSTM layers
        LSTM(50, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    return lstm_model


def train_arima_model(train):
    try:
        model = ARIMA(train, order=(10,1,0))
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        print("Error in ARIMA model training:", e)
        return None


def train_prophet_model(data):
    try:
        data = data.reset_index().rename(columns={'date': 'ds', 'transactions': 'y'})
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(data)
        return model
    except Exception as e:
        print("Error in Prophet model training:", e)
        return None


def train_lightgbm_model(X_train, y_train):
    try:
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print("Error in LightGBM model training:", e)
        return None


def train_model(model, X_train, y_train):
    try:
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[early_stop], shuffle=False)
    except Exception as e:
        print("Error in model training:", e)


if __name__ == "__main__":
    pass
