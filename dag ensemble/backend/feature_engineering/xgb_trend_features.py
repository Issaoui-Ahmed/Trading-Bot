import pandas as pd
from .shared import calculate_macd, calculate_rsi, calculate_sma

def engineer_features(df):
    df = df.copy()
    
    # Indicators
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['close'])
    df['ROC_10'] = df['close'].pct_change(10)
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    df['SMA_20'] = calculate_sma(df['close'], 20)
    
    return df
