import pandas as pd
import os


def save_all_forecasts(all_forecasts_dict, fname):
    df = pd.DataFrame()

    for name, info in all_forecasts_dict.items():
        fc = info.get('forecast')
        if fc is None:
            continue
        df[name] = fc

    df.to_csv(fname, index=True, date_format='%Y-%m-%d')
    print(f"[OK] Saved ALL forecasts → {fname}")


def save_forecast_single(name, forecast, outdir):
    fname = os.path.join(outdir, f"forecast_{name}.csv")

    df = pd.DataFrame({name: forecast})
    df.to_csv(fname, index=True, date_format='%Y-%m-%d')

    print(f"[OK] Saved forecast for {name} → {fname}")


def save_metrics(metrics_dict, fname):
    df = pd.DataFrame(metrics_dict).T
    df.to_csv(fname)
    print(f"[OK] Saved metrics → {fname}")
