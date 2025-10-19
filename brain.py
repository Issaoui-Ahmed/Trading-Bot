import pandas as pd
import time
import os
import uuid
import logging

from config import get_env, get_float_env, get_bool_env

META_SIGNALS_PATH = get_env("META_SIGNALS_PATH")
ACTIONS_PATH = get_env("ACTIONS_PATH")
PAIR = get_env("PAIR")
DEFAULT_VOLUME = get_float_env("DEFAULT_VOLUME")
ORDER_TYPE = get_env("ORDER_TYPE")
TIME_IN_FORCE = get_env("TIME_IN_FORCE")
VALIDATE = get_bool_env("VALIDATE")
LOG_PATH = get_env("ACTION_LOG_PATH")

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
