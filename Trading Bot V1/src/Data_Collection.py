import pandas as pd
import os
from datetime import datetime

# Get data to test on
df_path = "drive/MyDrive/Colab Notebooks/Trading Bot/Data/Crypto/SOL/"
combined_df = pd.DataFrame()
for file in os.listdir(df_path):
  df = pd.read_csv(df_path+file,header = None)
  df_columns = ['open_time', 'close_time', 'Open', 'High', 'Low', 'Close',
                'volume', 'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume', 'ignore', 'open_timestamp', 'close_timestamp']
  df.columns = ['open_timestamp','Open', 'High', 'Low', 'Close', 'volume', 'close_timestamp', 'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
  df["Date"] = df.open_timestamp.apply(lambda ts: datetime.fromtimestamp(ts/1000))
  df.index = df["Date"]
  df.sort_index(inplace = True)
  combined_df = pd.concat([combined_df,df])
  
combined_df = combined_df.drop(columns=["open_timestamp","close_timestamp","ignore","Date"])
  
combined_df.to_csv("drive/MyDrive/Colab Notebooks/Trading Bot/Data/Crypto/SOL/SOLUSDT-5m-2022-2023")
  
df = pd.read_csv("drive/MyDrive/Colab Notebooks/Trading Bot/Data/Crypto/SOL/SOLUSDT-5m-2022-2023")
df = df.iloc[:10000]

df.to_csv("drive/MyDrive/Colab Notebooks/Trading Bot/Data/Crypto/SOL/SOLUSDT-5m-test-sample")