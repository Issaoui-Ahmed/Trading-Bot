import requests
import pandas as pd
import time
import os

def download_kraken_data(pair='SOLUSD', interval=1):
    """
    Download OHLC data from Kraken public API.
    
    Args:
        pair (str): Kraken pair string (e.g., 'SOLUSD', 'XXBTZUSD').
        interval (int): Timeframe in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600).
    """
    url = f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}'
    
    print(f"Fetching data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data['error']:
            print(f"Error from Kraken API: {data['error']}")
            return

        # usage of result keys might vary, grabbing the first key's value which usually holds the data
        result = data['result']
        # The key might be 'SOLUSD' or 'XSOLZUSD' etc. just take the first one that is not 'last'
        keys = [k for k in result.keys() if k != 'last']
        if not keys:
            print("No data found in result keys.")
            return
            
        ohlc_list = result[keys[0]]
        
        # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
        columns = ['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        df = pd.DataFrame(ohlc_list, columns=columns)
        
        # Convert types
        df['time'] = pd.to_datetime(df['time'], unit='s')
        numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        output_file = os.path.join(os.path.dirname(__file__), 'kraken_solusd_1m.csv')
        df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}. Shape: {df.shape}")
        
    except Exception as e:
        print(f"Failed to download data: {e}")

if __name__ == "__main__":
    download_kraken_data()
