import pandas as pd
import lightgbm as lgb
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
LGB_MODEL_PATH = MODEL_DIR / "lgb_model.txt"
LGB_FEATURES_PATH = MODEL_DIR / "features_lgb.json"

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
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {"objective": "regression", "metric": "rmse"}
model = lgb.train(params, train_data)
model.save_model(LGB_MODEL_PATH)

with LGB_FEATURES_PATH.open("w") as f:
    json.dump(feature_cols, f)
