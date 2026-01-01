import pandas as pd
import numpy as np

def engineer_targets(df):
    """
    Calculates Meta Model classification targets (0=Buy, 1=Sell, 2=Hold)
    based on forward-looking price thresholds.
    """
    df = df.copy()
    
    # Calculate forward returns
    # standard float to handle pd.NA correctly
    df['close'] = df['close'].astype(float)
    
    current_closes = df['close']
    next_closes = df['close'].shift(-1)
    
    # Calculate returns: (next - current) / current
    returns = (next_closes - current_closes) / current_closes
    
    # Determine thresholds based on quantiles to ensure balanced classes
    # We want roughly 33% Buy, 33% Sell, 33% Hold
    
    # Drop NaNs for quantile calculation
    valid_returns = returns.dropna()
    
    if len(valid_returns) == 0:
        # Fallback if no data
        df['Target_Meta'] = 2
        return df
        
    lower_threshold = valid_returns.quantile(0.33)
    upper_threshold = valid_returns.quantile(0.66)
    
    # Vectorized calculation
    conditions = [
        (returns > upper_threshold), # Buy (Top 33%)
        (returns < lower_threshold)  # Sell (Bottom 33%)
    ]
    choices = [0, 1]
    
    # Default to 2 (Hold - Middle 33%)
    targets = np.select(conditions, choices, default=2)
    
    # Assign to DataFrame
    df['Target_Meta'] = targets
    
    # Handle the last row (NaN return) - usually becomes 0 or 2 depending on implementation of np.select with NaNs
    # explicitly set last row to Hold (2) or drop
    # For safety, we can leave it as computed (likely 2 because NaN > thresh is False)
    
    return df
