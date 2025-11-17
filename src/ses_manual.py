import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error

def ses_recursive(series, alpha, init=None):
    y = np.array(series)
    n = len(y)
    s = np.zeros(n)
    s[0] = y[0] if init is None else init
    for t in range(1, n):
        s[t] = alpha * y[t-1] + (1 - alpha) * s[t-1]
    return pd.Series(s, index=series.index)

def ses_forecast_last(train_series, alpha, steps):
    s = ses_recursive(train_series, alpha)
    last_level = s.iloc[-1]
    idx = pd.date_range(start=train_series.index[-1] + pd.offsets.DateOffset(months=1), periods=steps, freq=train_series.index.freq)
    return pd.Series([last_level]*steps, index=idx)

def grid_search_ses(train, val, alphas=np.linspace(0.01, 0.99, 99)):
    best_alpha, best_rmse, best_forecast = None, float('inf'), None
    for alpha in alphas:
        forecast = ses_forecast_last(train, alpha, len(val))
        rmse = sqrt(mean_squared_error(val.values, forecast.values))
        if rmse < best_rmse:
            best_alpha, best_rmse, best_forecast = alpha, rmse, forecast
    return best_alpha, best_rmse, best_forecast
