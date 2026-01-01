
import sys
import os
import pandas as pd
import pickle

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'backend')))

from models.xgb_meta_model import XGBMetaModel
from models.xgb_next_close import XGBNextClose
from models.xgb_trend import XGBTrend
from models.xgb_atr import XGBATR

def check_predictions():
    # Load Data
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'backend/datasets/SOL_USD_1m.csv'))
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")
    
    # Needs base model predictions first
    print("Generating base inputs...")
    model1 = XGBNextClose()
    model1.load()
    preds_next = model1.predict(df)
    
    model2 = XGBTrend()
    model2.load()
    preds_trend = model2.predict(df)
    
    model3 = XGBATR()
    model3.load()
    preds_atr = model3.predict(df)
    
    df_meta = df.copy()
    df_meta['xgb_next_close'] = preds_next
    df_meta['xgb_trend'] = preds_trend
    df_meta['xgb_atr'] = preds_atr
    
    # Predict with Meta Model
    print("Predicting with Meta Model...")
    meta_model = XGBMetaModel()
    meta_model.load()
    
    preds = meta_model.predict(df_meta)
    
    preds_series = pd.Series(preds, name='Meta_Preds')
    
    print("\nPrediction Distribution:")
    print(preds_series.value_counts(normalize=True))
    print("\nRaw Counts:")
    print(preds_series.value_counts())

if __name__ == "__main__":
    check_predictions()
