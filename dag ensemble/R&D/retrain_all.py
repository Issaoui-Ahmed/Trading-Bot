import pandas as pd
import os
import sys

# Setup paths to import models correctly
# We need to add 'backend/models' to path so we can import the classes
# But the model files themselves expect 'backend' to be in path to find 'feature_engineering'
# The model files add '..' relative to themselves to sys.path.

# Let's add backend/models to sys.path
backend_models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/models'))
sys.path.append(backend_models_path)

# We also need to ensure that when we import the model, it can find its dependencies.
# The model files contain: sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# This adds 'backend' to sys.path. Use index 0 to prioritize? No, append is fine.

try:
    from xgb_next_close import XGBNextClose
    from xgb_trend import XGBTrend
    from xgb_atr import XGBATR
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback/Debug: try adding backend explicitly if models fail to self-configure
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))
    from models.xgb_next_close import XGBNextClose
    from models.xgb_trend import XGBTrend
    from models.xgb_atr import XGBATR

def retrain_all():
    # Load Data
    # Use the larger dataset in backend/datasets if available
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/datasets/SOL_USD_1m.csv'))
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Train Model 1: Next Close
    print("\n--- Training Model 1: Next Close ---")
    model1 = XGBNextClose()
    model1.fit(df)

    # Train Model 2: Trend
    print("\n--- Training Model 2: Trend ---")
    model2 = XGBTrend()
    model2.fit(df)

    # Train Model 3: ATR
    print("\n--- Training Model 3: ATR ---")
    model3 = XGBATR()
    model3.fit(df)

    # Train Meta Model (Depends on others)
    print("\n--- Training Meta Model ---")
    from models.xgb_meta_model import XGBMetaModel
    
    # 1. Generate Base Model Predictions for Training Data
    print("Generating base model predictions for meta-training...")
    # Note: Using the just-trained models in-memory
    preds_next = model1.predict(df)
    preds_trend = model2.predict(df)
    preds_atr = model3.predict(df)
    
    # 2. Enrich DataFrame
    df_meta = df.copy()
    df_meta['xgb_next_close'] = preds_next
    df_meta['xgb_trend'] = preds_trend
    df_meta['xgb_atr'] = preds_atr
    
    # 3. Train Meta Model
    meta_model = XGBMetaModel()
    meta_model.fit(df_meta)

    print("\nAll models retrained and saved successfully.")

if __name__ == "__main__":
    retrain_all()
