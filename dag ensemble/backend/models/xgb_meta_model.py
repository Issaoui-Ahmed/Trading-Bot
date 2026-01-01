import xgboost as xgb
import pandas as pd
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from target_engineering.meta_model_targets import engineer_targets

class XGBMetaModel:
    def __init__(self):
        self.model_name = 'xgb_meta_model'
        # The features are the outputs of the base models
        self.features = ['xgb_atr', 'xgb_next_close', 'xgb_trend']
        self.target = 'Target_Meta'
        self.model = xgb.XGBClassifier(
            objective='multi:softprob', 
            num_class=3, 
            n_estimators=100, 
            eval_metric='mlogloss'
        )
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
        # 1. Feature Engineering: Assumes features (base model preds) are already in df
        # 2. Target Engineering (returns df with targets)
        df = engineer_targets(df) # Adds 'Target_Meta'
        
        # 3. Drop NaNs 
        data = df.dropna()
        
        X = data[self.features]
        y = data[self.target]
        
        print(f"Training {self.model_name} on {len(X)} rows...")
        self.model.fit(X, y)
        
        # Save
        os.makedirs(self.base_path, exist_ok=True)
        save_path = os.path.join(self.base_path, f'{self.model_name}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        print(f"Saved model to {save_path}")
        self.is_loaded = True

    def predict(self, df):
        # Assumes features are present
        X = df[self.features]
        
        if not self.is_loaded:
             self.load()
             if not self.is_loaded:
                 raise Exception("Model not loaded.")

        return self.model.predict(X)
