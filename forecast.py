def forecast_transactions(model_fit, steps):
    # Forecast future transactions
    forecast = model_fit.forecast(steps=steps)
    return forecast


def forecast_with_prophet(model, last_date, periods):
    # Create future dates of the specified periods
    future_dates = model.make_future_dataframe(periods=30, include_history=False)

    # Predict the values for the future dates
    forecast = model.predict(future_dates)

    # Filter the forecast to only include dates after the last date in the training data
    future_forecast = forecast[forecast['ds'] > last_date]  # use last_date directly

    return future_forecast[['ds', 'yhat']]  # 'yhat' is the forecasted value
