import requests
import pandas as pd
import time
import os

def fetch_data():
    # Binance Pair: SOLUSDT
    symbol = "SOLUSDT" 
    interval = "1m"
    
    # URL
    url = "https://api.binance.com/api/v3/klines"
    
    # Target: at least 15k bars (approx 10.4 days)
    target_bars = 15000
    
    # Start time: 15000 minutes ago
    # Multiplying by 60 for seconds, then * 1000 for milliseconds (Binance uses ms)
    start_time_ms = int((time.time() - 15000 * 60) * 1000)
    
    all_candles = []
    
    print(f"Fetching {symbol} {interval} data starting from approx {time.ctime(start_time_ms/1000)}...")
    
    current_start_ms = start_time_ms
    
    while len(all_candles) < target_bars:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start_ms,
            'limit': 1000
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Binance returns list of lists or error dict
            if isinstance(data, dict) and 'code' in data:
                print(f"Error: {data}")
                break
                
            if not data:
                print("No data returned")
                break
                
            candles = data
            first_ts = candles[0][0]
            last_ts = candles[-1][0]
            
            print(f"Fetched {len(candles)} candles. Range: {time.ctime(first_ts/1000)} to {time.ctime(last_ts/1000)}")
            
            all_candles.extend(candles)
            
            # Next start time is last close time + 1ms? 
            # Binance kline info:
            # [
            #   1499040000000,      // Open time
            #   "0.01634790",       // Open
            #   "0.80000000",       // High
            #   "0.01575800",       // Low
            #   "0.01577100",       // Close
            #   "148976.11427815",  // Volume
            #   1499644799999,      // Close time
            #   "2434.19055334",    // Quote asset volume
            #   308,                // Number of trades
            #   ...
            # ]
            
            # Start time for next batch should be > last open time.
            # Best is to use last Open Time + 60000ms? 
            # Or just take last candle Open Time and add 1 minute (60000 ms)
            
            last_open_time = candles[-1][0]
            current_start_ms = last_open_time + 60000
            
            # Check if we caught up to now
            if current_start_ms > time.time() * 1000:
                print("Caught up to present.")
                break
                
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Exception during fetch: {e}")
            break

    # Process into DataFrame
    # Target Cols: ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    processed_data = []
    
    for c in all_candles:
        ts = int(c[0] / 1000) # Convert to seconds for consistency with previous format
        o = float(c[1])
        h = float(c[2])
        l = float(c[3])
        cl = float(c[4])
        vol = float(c[5])
        q_vol = float(c[7])
        count = int(c[8])
        
        # VWAP = Quote Asset Volume / Volume
        if vol > 0:
            vwap = q_vol / vol
        else:
            vwap = cl # fallback
            
        processed_data.append([ts, o, h, l, cl, vwap, vol, count])
        
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    df = pd.DataFrame(processed_data, columns=cols)
    
    # Remove duplicates just in case
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "replay_datasets")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "SOL_USD_1m.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    fetch_data()
