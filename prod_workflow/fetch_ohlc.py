import requests
import pandas as pd

def fetch(pair, interval, since=None):
    url = "https://api.kraken.com/0/public/OHLC"
    r = requests.get(url, params={"pair": pair, "interval": interval, "since": since})
    d = r.json()
    if d["error"]:
        print("Kraken API Error:", d["error"])
        return None
    data = d["result"].get(pair, [])
    if not data:
        return pd.DataFrame()
    cols = ["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"]
    df = pd.DataFrame(data, columns=cols)
    df = df.astype({
        "timestamp": "int64",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "vwap": "float64",
        "volume": "float64",
        "count": "int64"
    })
    if since:
        return df[1:-1].reset_index(drop=True)
    return df[:-1].reset_index(drop=True)
