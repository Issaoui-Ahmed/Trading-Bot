import pandas as pd
import numpy as np
from .shared import calculate_atr, calculate_rsi, calculate_sma

def engineer_features(df):
    df = df.copy()
    
    # Volatility
    df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close'], 14)
    
    df['TR'] = pd.concat([df['high'] - df['low'], 
                          abs(df['high'] - df['close'].shift()), 
                          abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
                          
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Rolling_Std_20'] = df['close'].rolling(window=20).std()
    
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    df['SMA_20'] = calculate_sma(df['close'], 20)
    
    return df
