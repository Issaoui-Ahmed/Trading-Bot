import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import json
import time
import os

from config import get_env

SIGNALS_PATH = get_env("SIGNALS_PATH")
DATA_PATH = get_env("DATA_PATH")
XGB_MODEL_PATH = get_env("XGB_MODEL_PATH")
XGB_FEATURES_PATH = get_env("XGB_FEATURES_PATH")
LGB_MODEL_PATH = get_env("LGB_MODEL_PATH")
LGB_FEATURES_PATH = get_env("LGB_FEATURES_PATH")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(XGB_MODEL_PATH)
with open(XGB_FEATURES_PATH) as f:
    xgb_features = json.load(f)

lgb_model = lgb.Booster(model_file=LGB_MODEL_PATH)
with open(LGB_FEATURES_PATH) as f:
    lgb_features = json.load(f)

last_ts = None

while True:
    if not os.path.exists(DATA_PATH):
        time.sleep(5)
        continue

    df = pd.read_parquet(DATA_PATH)
    if "timestamp" not in df.columns or len(df) == 0:
        time.sleep(5)
        continue

    df = df.sort_values("timestamp").reset_index(drop=True)
    if last_ts is None:
        last_ts = df["timestamp"].iloc[-1]
        time.sleep(5)
        continue

    pending = df[df["timestamp"] > last_ts]
    if len(pending) == 0:
        time.sleep(5)
        continue

    pending = pending.dropna(subset=xgb_features + lgb_features)
    if len(pending) == 0:
        time.sleep(5)
        continue

    xgb_preds = xgb_model.predict(pending[xgb_features])
    lgb_preds = lgb_model.predict(pending[lgb_features])

    signals = pd.DataFrame({
        "timestamp": pending["timestamp"].values,
        "xgb_pred": xgb_preds,
        "lgb_pred": lgb_preds
    })

    signals.to_csv(SIGNALS_PATH, index=False)
    print(f"added to signals {len(pending)}")
    last_ts = pending["timestamp"].max()
    time.sleep(5)
