import pandas as pd

def engineer_targets(df):
    df = df.copy()
    # Ensure standard float to handle pd.NA correctly
    df['close'] = df['close'].astype(float)
    df['Target_Trend'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df
