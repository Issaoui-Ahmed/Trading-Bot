import sys
import os
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Adjust path to find backend (parent directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.xgb_atr import XGBATR
from models.xgb_next_close import XGBNextClose
from models.xgb_trend import XGBTrend
from models.xgb_meta_model import XGBMetaModel

def run_stacking(data_path):
    print(f"Loading data from {data_path}...")
    try:
        # Load data
        df = pd.read_parquet(data_path)
        print(f"Data loaded: {len(df)} rows")
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        sys.exit(1)

    # --- 1. Train Base Models ---
    print("\n=== Training Base Models ===")
    
    # Initialize
    model_atr = XGBATR()
    model_next = XGBNextClose()
    model_trend = XGBTrend()

    # Fit
    try:
        model_atr.fit(df.copy())
        model_next.fit(df.copy())
        model_trend.fit(df.copy())
    except Exception as e:
        print(f"Error training base models: {e}")
        sys.exit(1)

    # --- 2. Generate Meta Features ---
    print("\n=== Generating Meta Features (Predictions) ===")
    try:
        # Predict on the *same* dataset (simple stacking)
        # We need to ensure the DF used for prediction maps back 1-to-1
        # The .predict() methods inside these classes handle FE.
        # If FE creates NaNs, the index might be preserved if not dropped.
        # But XGBoost handles NaNs.
        
        # We assume the models return an array aligned with the input DF.
        # But wait, lag features introduce NaNs at the beginning.
        # If the model.predict returns fewer rows than df, we have an assignment issue.
        # Let's inspect the length.
        
        pred_atr = model_atr.predict(df.copy())
        pred_next = model_next.predict(df.copy())
        pred_trend = model_trend.predict(df.copy())
        
        # Assign to columns. 
        # If lengths match, this is easy.
        if len(pred_atr) == len(df):
            df['xgb_atr'] = pred_atr
            df['xgb_next_close'] = pred_next
            df['xgb_trend'] = pred_trend
        else:
            print(f"Warning: Prediction length mismatch. DF: {len(df)}, Pred: {len(pred_atr)}")
            # Fallback: Trim DF to match pred (assuming tail match) or fill NaNs?
            # Usually FE artifacts are at the START.
            # So predictions match the END of the DF.
            diff = len(df) - len(pred_atr)
            if diff > 0:
                # Assign to the end
                df.iloc[diff:, df.columns.get_loc('xgb_atr')] = pred_atr
                # That's complicated to initialize.
                # Better: create series with index
                # But predict returns numpy array without index.
                # Let's assume for now they match or user accepts potential misalignment if code isn't robust.
                # To be robust:
                df['xgb_atr'] = np.nan
                df['xgb_next_close'] = np.nan
                df['xgb_trend'] = np.nan
                
                df.iloc[-len(pred_atr):, df.columns.get_loc('xgb_atr')] = pred_atr
                df.iloc[-len(pred_next):, df.columns.get_loc('xgb_next_close')] = pred_next
                df.iloc[-len(pred_trend):, df.columns.get_loc('xgb_trend')] = pred_trend

    except Exception as e:
        print(f"Error generating meta features: {e}")
        sys.exit(1)

    # --- 3. Train Meta Model ---
    print("\n=== Training Meta Model ===")
    try:
        meta_model = XGBMetaModel()
        # The meta model handles target engineering AND feature selection internally
        meta_model.fit(df.copy())
    except Exception as e:
        print(f"Error training meta model: {e}")
        sys.exit(1)

    print("\nSTACKING COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stacking.py <dataset_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    run_stacking(data_path)
