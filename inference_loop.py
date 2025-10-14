import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()

SIGNALS_PATH = os.getenv("SIGNALS_PATH")
DATA_PATH = os.getenv("DATA_PATH")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("xgb_model.json")
with open("features.json") as f:
    xgb_features = json.load(f)

lgb_model = lgb.Booster(model_file="lgb_model.txt")
with open("features_lgb.json") as f:
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
