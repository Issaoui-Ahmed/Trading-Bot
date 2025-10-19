import pandas as pd
import time
import os
from datetime import datetime, UTC

from fetch_ohlc import fetch
from feature_creation import create_features
from config import get_env, get_int_env

FILE_PATH = get_env("DATA_PATH")
MAX_ROWS = get_int_env("MAX_ROWS")

PAIR = get_env("PAIR")
INTERVAL = get_int_env("INTERVAL")


def initialize_parquet():
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)
    df = fetch(PAIR, INTERVAL)
    feats = create_features(df)
    df = pd.concat([df, feats], axis=1)
    df.to_parquet(FILE_PATH, index=False)
    print("initialized data")

def incremental_update_loop():
    df = pd.read_parquet(FILE_PATH)
    while True:
        last_ts = int(df["timestamp"].astype(float).max())
        last_minute = datetime.fromtimestamp(last_ts, UTC).minute
        current_minute = datetime.now(UTC).minute

        if current_minute != (last_minute + 1) % 60:
            new_data = fetch(PAIR, INTERVAL,since=int(last_ts))
            if new_data is not None and not new_data.empty:
                df = pd.concat([df, new_data], ignore_index=True)
                df = df.tail(MAX_ROWS)  # keep only the last 5000 rows
                feats = create_features(df)
                df = pd.concat([df.iloc[:, :len(new_data.columns)], feats], axis=1)
                df.to_parquet(FILE_PATH, index=False)
                print("incremented data")
            time.sleep(5)
        else:
            time.sleep(1)

if __name__ == "__main__":
    initialize_parquet()
    incremental_update_loop()
