import os
import time
import hmac
import hashlib
import base64
import requests
import pandas as pd
import logging
from urllib.parse import urlencode

from _utils.config import get_env

ACTIONS_PATH = get_env("ACTIONS_PATH")
LOG_PATH = get_env("BROKER_LOG_PATH")
KRAKEN_API_KEY = get_env("KRAKEN_API_KEY")
KRAKEN_API_SECRET = get_env("KRAKEN_API_SECRET")
KRAKEN_BASE_URL = "https://api.kraken.com"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

if not KRAKEN_API_KEY or not KRAKEN_API_SECRET:
    raise ValueError(
        "Missing Kraken API credentials. Set KRAKEN_API_KEY and "
        "KRAKEN_API_SECRET in your .env.key file."
    )

last_ts = None
logging.info("Broker started — watching for new actions.")

def kraken_signature(url_path, data, secret):
    postdata = urlencode(data)
    encoded = (str(data['nonce']) + postdata).encode()
    message = url_path.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
    sigdigest = base64.b64encode(mac.digest())
    return sigdigest.decode()

def send_order(order):
    endpoint = "/0/private/AddOrder"
    url = KRAKEN_BASE_URL + endpoint

    data = {
        "nonce": order["nonce"],
        "ordertype": order["ordertype"],
        "type": order["side"],
        "volume": str(order["volume"]),
        "pair": order["pair"],
        "validate": str(order["validate"]).lower(),
        "timeinforce": order.get("time_in_force", "gtc")
    }

    if order.get("price"):
        data["price"] = str(order["price"])
    if order.get("price2"):
        data["price2"] = str(order["price2"])
    if order.get("expiretm"):
        data["expiretm"] = str(order["expiretm"])
    if order.get("userref"):
        data["userref"] = order["userref"]

    headers = {
        "API-Key": KRAKEN_API_KEY,
        "API-Sign": kraken_signature(endpoint, data, KRAKEN_API_SECRET)
    }

    try:
        r = requests.post(url, headers=headers, data=data, timeout=10)
        res = r.json()

        if res.get("error"):
            logging.error(f"Order failed ({order['side']} {order['pair']}): {res['error']}")
        else:
            logging.info(f"Order successful: {res['result']}")
        return res

    except Exception as e:
        logging.error(f"Order error for {order}: {e}", exc_info=True)
        return None

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
            logging.info(f"Initialized last timestamp to {last_ts}.")
            time.sleep(5)
            continue

        pending = df[df["timestamp"] > last_ts]
        if len(pending) == 0:
            time.sleep(5)
            continue

        for _, row in pending.iterrows():
            order = row.to_dict()
            res = send_order(order)
            if res:
                print(f"Executed: {order['side']} {order['pair']} | result: {res}")
            time.sleep(2)  # avoid hitting Kraken rate limits

        last_ts = pending["timestamp"].max()
        logging.info(f"Processed {len(pending)} new actions up to {last_ts}.")

        time.sleep(5)

    except Exception as e:
        logging.error(f"Broker loop error: {e}", exc_info=True)
        time.sleep(5)
