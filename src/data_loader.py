import pandas as pd
import statsmodels.api as sm

DATE_FREQ = 'MS'

def load_airpassengers():
    data = sm.datasets.get_rdataset("AirPassengers").data
    s = data['value'] if 'value' in data.columns else data['Passengers']
    idx = pd.date_range(start='1949-01-01', periods=len(s), freq=DATE_FREQ)
    ts = pd.Series(s.values.astype(float), index=idx, name='value')
    return ts

def load_from_csv(path, date_col, value_col, freq=DATE_FREQ):
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col)
    ts = pd.Series(df[value_col].values, index=pd.DatetimeIndex(df[date_col]), name=value_col)
    return ts.asfreq(freq)
