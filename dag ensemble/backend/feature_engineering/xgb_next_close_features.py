import pandas as pd
from .shared import calculate_rsi, calculate_sma

def engineer_features(df):
    df = df.copy()
    
    # Common
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    df['SMA_20'] = calculate_sma(df['close'], 20)
    
    # Lags
    for lag in [1, 2, 3]:
        df[f'Close_Lag{lag}'] = df['close'].shift(lag)
        
    return df
