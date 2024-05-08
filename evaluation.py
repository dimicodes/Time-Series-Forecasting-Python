import os
from prophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models import load_and_preprocess_data, train_arima_model, train_prophet_model, build_nn_model, build_lstm_model
from models import train_lightgbm_model
from forecast import *


def evaluate_models(actual, forecasts):
    evaluation_results = {}

    for model_name, forecast in forecasts.items():
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(actual, forecast)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(actual, forecast)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Calculate R-Squared (R2) Score
        r2 = r2_score(actual, forecast)

        evaluation_results[model_name] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2 Score': r2}

    return evaluation_results


def main():
    filepath = '/Users/deikaplan/Downloads/Forecasting-Python/datasets/forecasting_data.csv'
    data = load_and_preprocess_data(filepath)
    # data = data.sort_index()  # sort data chronologically

    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    train = train.fillna(0)

    # Convert datetime index to numeric
    train_index_numeric = np.arange(len(train))
    test_index_numeric = np.arange(len(train), len(train) + len(test))




    # Training the models

    # Train Prophet Model
    # Prepare data for Prophet
    train_prophet = train.reset_index().rename(columns={'date': 'ds', 'transactions': 'y'})

    # Train the Prophet model on all available data
    prophet_model = Prophet()
    prophet_model.fit(train_prophet)

    # Forecast beyond the last training date
    future = prophet_model.make_future_dataframe(periods=len(test) + 180)
    forecast = prophet_model.predict(future)

    # Forecast for model evaluation
    prophet_forecast = prophet_model.predict(test.reset_index().rename(columns={'date': 'ds'}))

    # Plot the forecast
    fig1 = plot_plotly(prophet_model, forecast)
    fig1.show()


    # Train ARIMA Model
    arima_model = train_arima_model(train['transactions'])
    arima_forecast = arima_model.forecast(steps=len(test))
    arima_forecast_series = pd.Series(arima_forecast.values, index=test.index)

    # Train NN Model
    nn_model = build_nn_model(1)  # Adjust input shape
    nn_model.fit(train_index_numeric.reshape(-1, 1), train.values, epochs=100, batch_size=1, verbose=0)
    nn_forecast = nn_model.predict(test_index_numeric.reshape(-1, 1))
    nn_forecast_series = pd.Series(nn_forecast.flatten(), index=test.index)

    # Train LSTM Model
    lstm_model = build_lstm_model(1)  # Adjust input shape
    lstm_model.fit(train_index_numeric.reshape(-1, 1), train.values, epochs=100, batch_size=1, verbose=0)
    lstm_forecast = lstm_model.predict(test.values.reshape((test.shape[0], 1, 1)))
    lstm_forecast_series = pd.Series(lstm_forecast.flatten(), index=test.index)

    # Train LightGBM Model
    lightgbm_model = train_lightgbm_model(train_index_numeric.reshape(-1, 1), train['transactions'])
    lightgbm_forecast = lightgbm_model.predict(test_index_numeric.reshape(-1, 1))
    lightgbm_forecast_series = pd.Series(lightgbm_forecast, index=test.index)

    # Evaluate Models
    evaluation_results = evaluate_models(test['transactions'], {'ARIMA': arima_forecast_series, 'Prophet': prophet_forecast['yhat'],
                                                             'NN': nn_forecast_series, 'LSTM': lstm_forecast_series,
                                                             'LightGBM': lightgbm_forecast_series})
    evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
    print("\nEvaluation Metrics:")
    print(evaluation_df)




    # Train Prophet and get forecasted transaction volumes

    # Train the Prophet model on all available data
    prophet_model = train_prophet_model(data.reset_index().rename(columns={'date': 'ds', 'transactions': 'y'}))

    # Get the last date from the full data set
    last_date = data.index.max()
    print("\nLast date in dataset:", last_date)

    # Ensure you are forecasting beyond this last date
    future_transactions = forecast_with_prophet(prophet_model, last_date, 30)
    print("\nFuture Transactions Forecast:")
    print(future_transactions)






if __name__ == "__main__":
    main()
