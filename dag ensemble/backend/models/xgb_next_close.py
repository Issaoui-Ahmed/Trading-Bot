import xgboost as xgb
import pandas as pd
import os
import sys
import pickle

# Adjust path to find backend (parent directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ModelWrapper removed

from feature_engineering.xgb_next_close_features import engineer_features
from target_engineering.xgb_next_close_targets import engineer_targets

class XGBNextClose:
    def __init__(self):
        self.model_name = 'xgb_next_close'
        self.features = ['close', 'open', 'high', 'low', 'volume', 'Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'RSI_14', 'SMA_20']
        self.target = 'Target_Next_Close'
        self.model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'backend', 'pretrained_models')
        self.is_loaded = False

    def load(self):
        model_path = os.path.join(self.base_path, f'{self.model_name}.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_loaded = True
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model not found at {model_path}")

    def fit(self, df):
        # 1. Feature Engineering
        df = engineer_features(df)
        
        # 2. Target Engineering
        df = engineer_targets(df)
        
        # 3. Drop NaNs created by lags/rolling
        df_clean = df.dropna()
        
        X = df_clean[self.features]
        y = df_clean[self.target]
        
        print(f"Training {self.model_name}...")
        self.model.fit(X, y)
        
        # Save
        os.makedirs(self.base_path, exist_ok=True)
        save_path = os.path.join(self.base_path, f'{self.model_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Saved model to {save_path}")
        self.is_loaded = True

    def predict(self, df):
        # 1. Feature Engineering
        df = engineer_features(df)
        
        # 2. Prepare X
        X = df[self.features]
        
        # Ensure loaded
        if not self.is_loaded:
             self.load()
             if not self.is_loaded:
                 raise Exception("Model not loaded and valid model file not found.")

        # Native XGB predict
        return self.model.predict(X)
