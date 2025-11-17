from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_statsmodels_models(train, val, seasonal_periods=12):
    results = {}
    try:
        ses_model = ExponentialSmoothing(train, trend=None, seasonal=None, initialization_method='estimated')
        ses_fit = ses_model.fit(optimized=True)
        results['SES'] = {'fit': ses_fit, 'forecast': ses_fit.forecast(len(val))}
    except:
        results['SES'] = {'fit': None, 'forecast': None}

    try:
        holt_model = ExponentialSmoothing(train, trend='add', seasonal=None, initialization_method='estimated')
        holt_fit = holt_model.fit(optimized=True)
        results['Holt'] = {'fit': holt_fit, 'forecast': holt_fit.forecast(len(val))}
    except:
        results['Holt'] = {'fit': None, 'forecast': None}

    try:
        hw_add = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_periods, initialization_method='estimated')
        hw_add_fit = hw_add.fit(optimized=True)
        results['HW_add'] = {'fit': hw_add_fit, 'forecast': hw_add_fit.forecast(len(val))}
    except:
        results['HW_add'] = {'fit': None, 'forecast': None}

    try:
        hw_mul = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=seasonal_periods, initialization_method='estimated')
        hw_mul_fit = hw_mul.fit(optimized=True)
        results['HW_mul'] = {'fit': hw_mul_fit, 'forecast': hw_mul_fit.forecast(len(val))}
    except:
        results['HW_mul'] = {'fit': None, 'forecast': None}

    return results
