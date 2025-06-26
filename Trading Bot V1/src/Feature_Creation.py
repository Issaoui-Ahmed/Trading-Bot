import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas_ta as ta

import ML_usage as mlu
import Task_Evaluation as te
import Feature_Eval as fev

from importlib import reload
def reload_modules():
    for module in [fev, te, mlu]:
        reload(module) 

df = pd.read_csv("../Data/SOLUSDT-5m-2022-2023-labeled.csv")
df.index = pd.to_datetime(df["Date"])
df.drop(columns = ["Date"],inplace = True)
df.sort_index(inplace = True)
df.index = pd.to_datetime(df.index)

 
OHLC = ["Open","Close","High","Low"]
periods = [5,14,50,100]
def add_distances(feature,df):
   for price in OHLC:
        df[f"Absolute_{price}-{feature}"] = df[price] - df[f"Absolute_{feature}"]
        df[f"RelativeCL_{price}-{feature}"] = df[f"Absolute_{price}-{feature}"] / df["Close"]
        df[f"RelativeATR10_{price}-{feature}"] = df[f"Absolute_{price}-{feature}"] / df[f'Absolute_ATR_10']
        df[f"RelativeATR20_{price}-{feature}"] = df[f"Absolute_{price}-{feature}"] / df[f'Absolute_ATR_20']
def add_relative_features(feature,p,df):
  df[f'RelativeCL_{feature}'] = df[f'Absolute_{feature}'] / df["Close"]
  df[f'RelativeATR{p}_{feature}'] = df[f'Absolute_{feature}'] / df[f'Absolute_ATR_{p}']
def add_line_features(line,name,p,df):
    df[f"above_{name}_{p}"] = np.where(df['Low'] > line, 1, 0)
    df[f"below_{name}_{p}"] = np.where(df['High'] < line, 1, 0)
    df[f"through_{name}_{p}"] = np.where((df[f"above_{name}_{p}"] == 0) & (df[f"below_{name}_{p}"] == 0), 1, 0)
def add_moving_averages(df):
    for p in periods:
        ma = df['Close'].rolling(window=p).mean()
        weights = np.arange(1, p+1)
        wma = df['Close'].rolling(window=p).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
        ema = df['Close'].ewm(span=p, adjust=False).mean()
        ema2 = ema.ewm(span=5, adjust=False).mean()
        ema3 = ema2.ewm(span=5, adjust=False).mean()
        tema = 3 * ema - 3 * ema2 + ema3
        moving_averages = [ma,wma,ema,ema2,ema3,tema]
        moving_averages_names = ["ma","wma","ema","ema2","ema3","tema"]
        for line,name in zip(moving_averages,moving_averages_names):
           add_line_features(line,name,p,df)

        # df[f'Absolute_MA_Close_{p}'] = df['Close'].rolling(window=p).mean()
        # weights = np.arange(1, p+1)
        # df[f'Absolute_WMA_Close_{p}'] = df['Close'].rolling(window=p).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
        # df[f'Absolute_EMA_Close_{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
        # ema2 = df[f'Absolute_EMA_Close_{p}'].ewm(span=5, adjust=False).mean()
        # ema3 = ema2.ewm(span=5, adjust=False).mean()
        # df[f'Absolute_TEMA_Close_{p}'] = 3 * df[f'Absolute_EMA_Close_{p}'] - 3 * ema2 + ema3
        # moving_averages = [f"MA_Close_{p}",f"WMA_Close_{p}",f"EMA_Close_{p}",f"TEMA_Close_{p}"]
        # for ma in moving_averages:
        #   add_relative_features(ma,p,df)
        #   add_distances(ma, df)
    return df
def add_ATR(df):
   for p in periods:
      df[f'Absolute_ATR_{p}'] = ta.atr(df['High'], df['Low'], df['Close'],length = p)
      df[f'Relative_ATR_{p}'] = df[f'Absolute_ATR_{p}'] / df["Close"]
   return df
def preprocess_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    df = df[df['Close'] != 0]
    df = df[df['Open'] != 0]
    return df
def extract_features(df):
   df_extracted = df.copy()
  #  df = add_ATR(df)
   df_extracted = add_moving_averages(df_extracted)
   df_extracted = preprocess_data(df_extracted)
   return df_extracted


fev.test_covariate_shift(df.Open.pct_change())

df_extracted = extract_features(df)

 
X = df.drop(['Reversal'], axis=1)
y = df["Reversal"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,shuffle = False)

reload_modules()
model, y_pred = fev.asses_all_features(X_train, X_test, y_train, y_test)

## idea: predict supertrend signal in next candle

 
long_positions, short_positions = mlu.exc(pd.Series(y_pred), df.loc[y_test.index])

 
te.exc(long_positions, short_positions, df.Open)


