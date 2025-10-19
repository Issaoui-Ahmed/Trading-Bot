import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

IGNORED_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "vwap",
    "volume",
    "count",
    "Target",
}

DATA_PATH = Path("prod_data/solusd_data_feats.parquet")
MODEL_DIR = Path("prod_models")
XGB_MODEL_PATH = MODEL_DIR / "xgb_model.json"
XGB_FEATURES_PATH = MODEL_DIR / "features.json"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df["Target"] = df["close"].shift(-1)
df = df.dropna()

feature_cols = [c for c in df.columns if c not in IGNORED_COLUMNS]
X = df[feature_cols]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
model.save_model(XGB_MODEL_PATH)

with XGB_FEATURES_PATH.open("w") as f:
    json.dump(feature_cols, f)
