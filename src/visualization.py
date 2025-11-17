import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set(style="whitegrid")  # стиль графіків

def plot_series_with_forecasts(train, val, test, forecasts_dict, title='Series and forecasts', fname=None):
    plt.figure(figsize=(12,6))
    plt.plot(train, label='Train', marker='o')
    plt.plot(val, label='Validation', marker='o')
    plt.plot(test, label='Test', marker='o')
    for name, info in forecasts_dict.items():
        fc = info.get('forecast')
        if fc is not None:
            plt.plot(fc, label=f'Forecast: {name}', linestyle='--', marker='x')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_residuals(actual, forecast, title='Residuals', fname=None):
    residuals = actual - forecast
    plt.figure(figsize=(12,6))
    plt.plot(residuals, marker='o')
    plt.axhline(0, color='black', linewidth=1)
    plt.title(f'{title}')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.grid(True)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


def decompose_series(ts, model='multiplicative', period=12, fname=None):
    decomp = seasonal_decompose(ts, model=model, period=period, extrapolate_trend='freq')
    fig = decomp.plot()
    fig.set_size_inches(12,8)
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    plt.show()
    return decomp
