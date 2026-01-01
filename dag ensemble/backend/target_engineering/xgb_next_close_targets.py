import pandas as pd

def engineer_targets(df):
    df = df.copy()
    df['Target_Next_Close'] = df['close'].shift(-1)
    return df
