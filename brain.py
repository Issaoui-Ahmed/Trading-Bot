import pandas as pd
import time
import os
import json
import uuid
import logging
from dotenv import load_dotenv

load_dotenv()

META_SIGNALS_PATH = os.getenv("META_SIGNALS_PATH", "meta_signals.csv")
ACTIONS_PATH = os.getenv("ACTIONS_PATH", "actions.csv")
PAIR = os.getenv("PAIR", "SOLUSD")
DEFAULT_VOLUME = float(os.getenv("DEFAULT_VOLUME", 1.0))
ORDER_TYPE = os.getenv("ORDER_TYPE", "market")
TIME_IN_FORCE = os.getenv("TIME_IN_FORCE", "gtc")
VALIDATE = os.getenv("VALIDATE", "false").lower() == "true"
LOG_PATH = os.getenv("ACTION_LOG_PATH", "action.log")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

last_ts = None
logging.info("Action generation loop started.")

signal_to_side = {
    "enter_long": "buy",
    "enter_short": "sell",
    "exit_long": "sell",
    "exit_short": "buy",
    "none": None
}

def generate_order(row):
    signal = row["meta_signal"]
    side = signal_to_side.get(signal, None)
    if side is None:
        return None

    ts = int(row["timestamp"])
    nonce = int(time.time() * 1000)
    userref = str(uuid.uuid4())

    order = {
        "timestamp": ts,
        "pair": PAIR,
        "side": side,
        "ordertype": ORDER_TYPE,
        "volume": DEFAULT_VOLUME,
        "price": None,
        "price2": None,
        "time_in_force": TIME_IN_FORCE,
        "expiretm": None,
        "userref": userref,
        "validate": VALIDATE,
        "nonce": nonce,
        "signal": signal
    }
    return order

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

        orders = []
        for _, row in pending.iterrows():
            order = generate_order(row)
            if order:
                orders.append(order)

        if orders:
            orders_df = pd.DataFrame(orders)
            orders_df.to_csv(ACTIONS_PATH, index=False)

            last_ts = pending["timestamp"].max()
            logging.info(f"Wrote {len(orders)} new actions up to {last_ts}.")

        time.sleep(5)

    except Exception as e:
        logging.error(f"Error in action loop: {e}", exc_info=True)
        time.sleep(5)
