import pandas as pd

def engineer_targets(df):
    df = df.copy()
    # Assumes ATR_14 is available (computed in feature engineering)
    if 'ATR_14' in df.columns:
        df['Target_Next_ATR'] = df['ATR_14'].shift(-1)
    else:
        # Fallback if not present, though it should be
        # We can re-import calculate_atr if strictly needed, but relying on FE is cleaner based on flow
        raise ValueError("ATR_14 feature missing. Ensure feature engineering is run before target engineering.")
    return df
