from sklearn.metrics import mean_absolute_error
from math import sqrt

def evaluate_models(actual, forecasts_dict):
    scores = {}
    for name, info in forecasts_dict.items():
        fc = info.get('forecast')
        if fc is None:
            scores[name] = {'MAE': None, 'RMSE': None}
            continue
        scores[name] = {
            'MAE': mean_absolute_error(actual.values, fc.values),
            'RMSE': sqrt(((actual.values - fc.values)**2).mean())
        }
    return scores
