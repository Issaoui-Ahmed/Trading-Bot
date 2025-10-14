import pandas as pd
import xgboost as xgb
import json
import time
import os
import numpy as np
import logging

SIGNALS_PATH = "signals.csv"
META_SIGNALS_PATH = "meta_signals.csv"
META_MODEL_PATH = "meta_model.json"
META_FEATURES_PATH = "meta_features.json"
META_LABELS_PATH = "meta_labels.json"
LOG_PATH = "meta_inference.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

meta_model = xgb.XGBClassifier()
meta_model.load_model(META_MODEL_PATH)

with open(META_FEATURES_PATH) as f:
    meta_features = json.load(f)
with open(META_LABELS_PATH) as f:
    meta_labels = json.load(f)

actions = ["enter_long", "enter_short", "exit_long", "exit_short", "none"]
last_ts = None

logging.info("Meta inference loop started.")

while True:
    try:
        if not os.path.exists(SIGNALS_PATH):
            time.sleep(5)
            continue

        df = pd.read_csv(SIGNALS_PATH)
        if "timestamp" not in df.columns or len(df) == 0:
            time.sleep(5)
            continue

        df = df.sort_values("timestamp").reset_index(drop=True)
        if last_ts is None:
            last_ts = df["timestamp"].iloc[-1]
            logging.info(f"Initialized last timestamp to {last_ts}.")
            time.sleep(5)
            continue

        pending = df[df["timestamp"] > last_ts]
        if len(pending) == 0:
            time.sleep(5)
            continue

        pending = pending.dropna(subset=meta_features)
        if len(pending) == 0:
            time.sleep(5)
            continue

        X_meta = pending[meta_features]
        preds = meta_model.predict(X_meta)

        meta_signals = pd.DataFrame({
            "timestamp": pending["timestamp"].values,
            "meta_signal": np.random.choice(actions, len(pending))  # random for now
        })

        if os.path.exists(META_SIGNALS_PATH):
            meta_signals.to_csv(META_SIGNALS_PATH, mode="a", header=False, index=False)
        else:
            meta_signals.to_csv(META_SIGNALS_PATH, index=False)

        last_ts = pending["timestamp"].max()
        logging.info(f"Appended {len(meta_signals)} meta-signals up to {last_ts}.")

        time.sleep(5)

    except Exception as e:
        logging.error(f"Meta inference error: {e}", exc_info=True)
        time.sleep(5)
