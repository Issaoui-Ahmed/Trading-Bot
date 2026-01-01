import xgboost as xgb
import pandas as pd
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ModelWrapper removed

from feature_engineering.xgb_atr_features import engineer_features
from target_engineering.xgb_atr_targets import engineer_targets

class XGBATR:
    def __init__(self):
        self.model_name = 'xgb_atr'
        self.features = ['ATR_14', 'TR', 'High_Low_Diff', 'Rolling_Std_20', 'RSI_14', 'SMA_20']
        self.target = 'Target_Next_ATR'
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
        df = engineer_features(df)
        df = engineer_targets(df)
        df_clean = df.dropna()
        
        X = df_clean[self.features]
        y = df_clean[self.target]
        
        print(f"Training {self.model_name}...")
        self.model.fit(X, y)
        
        os.makedirs(self.base_path, exist_ok=True)
        save_path = os.path.join(self.base_path, f'{self.model_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        print(f"Saved model to {save_path}")
        self.is_loaded = True

    def predict(self, df):
        df = engineer_features(df)
        X = df[self.features]
        
        if not self.is_loaded:
             self.load()
             if not self.is_loaded:
                 raise Exception("Model not loaded.")

        return self.model.predict(X)
