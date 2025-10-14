import pandas as pd
import time
import os
import logging

META_SIGNALS_PATH = "meta_signals.csv"
ACTIONS_PATH = "actions.csv"
LOG_PATH = "brain.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

last_ts = None
logging.info("Brain loop started.")

while True:
    try:
        if not os.path.exists(META_SIGNALS_PATH):
            time.sleep(5)
            continue

        df = pd.read_csv(META_SIGNALS_PATH)
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

        actions = pd.DataFrame({
            "timestamp": pending["timestamp"].values,
            "decision": pending["meta_signal"].values
        })

        if os.path.exists(ACTIONS_PATH):
            actions.to_csv(ACTIONS_PATH, mode="a", header=False, index=False)
        else:
            actions.to_csv(ACTIONS_PATH, index=False)

        last_ts = pending["timestamp"].max()
        logging.info(f"Wrote {len(actions)} new actions up to {last_ts}.")

        time.sleep(5)

    except Exception as e:
        logging.error(f"Brain loop error: {e}", exc_info=True)
        time.sleep(5)
