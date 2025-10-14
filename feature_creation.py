import pandas as pd

def ema(data, period=10):
    return data['close'].ewm(span=period, adjust=False).mean()

def open_minus_close(data):
    return data['open'] - data['close']

def create_features(data):
    ema_feature = ema(data)
    diff_feature = open_minus_close(data)
    return pd.DataFrame({
        'EMA': ema_feature,
        'Open_Close_Diff': diff_feature
    })
