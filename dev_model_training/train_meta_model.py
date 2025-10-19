import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

DATA_PATH = Path("prod_data/signals.csv")
MODEL_DIR = Path("prod_models")
META_MODEL_PATH = MODEL_DIR / "meta_model.json"
META_FEATURES_PATH = MODEL_DIR / "meta_features.json"
META_LABELS_PATH = MODEL_DIR / "meta_labels.json"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Meta training data not found at {DATA_PATH}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna()

df["label"] = np.random.choice(["enter_long","enter_short","exit_long","exit_short","none"], size=len(df))

X = df[["xgb_pred", "lgb_pred"]]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=True)

meta_model = xgb.XGBClassifier()
meta_model.fit(X_train, y_train)

meta_model.save_model(META_MODEL_PATH)

with META_FEATURES_PATH.open("w") as f:
    json.dump(list(X.columns), f)

with META_LABELS_PATH.open("w") as f:
    json.dump(list(le.classes_), f)
