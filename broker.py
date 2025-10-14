import pandas as pd
import time
import os
import logging
import requests

ACTIONS_PATH = "actions.csv"
LOG_PATH = "broker.log"

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def execute_order(signal):
    # TODO: replace with actual Kraken order logic
    logging.info(f"Executing order: {signal}")
    print(f"Executing order: {signal}")
    # Example placeholder:
    # requests.post("https://api.kraken.com/0/private/AddOrder", headers=auth_headers, data=payload)

last_ts = None
logging.info("Broker loop started.")

while True:
    try:
        if not os.path.exists(ACTIONS_PATH):
            time.sleep(5)
            continue

        df = pd.read_csv(ACTIONS_PATH)
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

        for _, row in pending.iterrows():
            signal = row["decision"]
            execute_order(signal)

        last_ts = pending["timestamp"].max()
        logging.info(f"Executed {len(pending)} actions up to {last_ts}.")

        time.sleep(5)

    except Exception as e:
        logging.error(f"Broker loop error: {e}", exc_info=True)
        time.sleep(5)
