import requests
import time
import pandas as pd
from datetime import datetime, timedelta

class KrakenFeed:
    """
    A simple wrapper for the Kraken public API to fetch OHLC data.
    """
    def __init__(self):
        self.base_url = "https://api.kraken.com/0/public/OHLC"
        self.interval_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080,
            '15d': 21600
        }

    def fetch_ohlcv(self, pair: str, timeframe: str):
        """
        Fetches the latest OHLCV data for a given pair and timeframe.
        
        Args:
            pair (str): The trading pair, e.g., 'XBTUSD', 'ETHUSD'. 
                        Note: Kraken uses slightly different symbols (e.g. XXBTZUSD), 
                        but usually handles query aliases well.
            timeframe (str): The time interval, e.g., '1m', '1h'.
            
        Returns:
            dict: The latest candle data or None if failed.
        """
        interval = self.interval_map.get(timeframe)
        if not interval:
            print(f"Error: Invalid timeframe '{timeframe}'")
            return None

        try:
            # We need enough history for feature engineering (e.g., rolling windows)
            # Fetching 720 candles ensures we cover most standard lookback periods (RSI 14, SMA 200, etc.)
            params = {
                'pair': pair.replace("/", ""),
                'interval': interval,
                'since': int(time.time() - interval * 60 * 720) 
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('error'):
                print(f"Kraken API Error for {pair}: {data['error']}")
                return None

            result = data.get('result')
            if not result:
                return None

            # The key for the pair data might be dynamic
            pair_key = [k for k in result.keys() if k != 'last'][0]
            candles = result[pair_key]

            if not candles:
                return None

            # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            df = pd.DataFrame(candles, columns=cols)
            
            # Convert types
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = df[c].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df

        except Exception as e:
            print(f"Exception fetching Kraken data for {pair}: {e}")
            return None

if __name__ == "__main__":
    # Test
    feed = KrakenFeed()
    print(feed.fetch_ohlcv("BTC/USD", "1m"))
