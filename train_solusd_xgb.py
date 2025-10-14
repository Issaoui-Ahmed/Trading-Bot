import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import json

df = pd.read_parquet("ohlcv.parquet")
df["Target"] = df["close"].shift(-1)
df = df.dropna()

feature_cols = [c for c in df.columns if c not in ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count", "Target"]]
X = df[feature_cols]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
model.save_model("xgb_model.json")

with open("features.json", "w") as f:
    json.dump(feature_cols, f)
