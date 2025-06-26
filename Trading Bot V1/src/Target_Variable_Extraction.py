
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
from scipy.signal import savgol_filter , find_peaks
import pandas_ta as ta
import math
from collections import Counter

df = pd.read_csv("drive/MyDrive/Colab Notebooks/Trading Bot/Data/Crypto/SOL/SOLUSDT-5m-2022-2023")
df.index = pd.to_datetime(df["Date"])
df.drop(columns = ["Date"],inplace = True)
df.sort_index(inplace = True)

factor_1 = 0.7
factor_2 = 0.15
def before_up_mask(points,avg_candle,factor_1 = factor_1, factor_2 = factor_2):
   return ~( ((points) >= (points.shift(1) + (avg_candle * factor_1)))
          | ( (points.shift(1) < (points - avg_candle * factor_2)) &  (points < (points.shift(-1) - avg_candle * factor_2)) ) )
def next_up_mask(points,avg_candle,factor_1 = factor_1, factor_2 = factor_2):
   return   ( ((points.shift(-1)) >= (points + (avg_candle * factor_1)))
              | ( (points < (points.shift(-1) - avg_candle * factor_2)) &  (points.shift(-1) < (points.shift(-2) - avg_candle * factor_2)) ) )
def is_up(points,avg_candle,factor_1 = factor_1 , factor_2 = factor_2):
  #  return next_up_mask(points,avg_candle) & before_up_mask(points,avg_candle)
   return next_up_mask(points,avg_candle)
def before_down_mask(points,avg_candle,factor_1 = factor_1, factor_2 = factor_2):
   return ~( ((points) <= (points.shift(1) - (avg_candle * factor_1)))
          | ( (points.shift(1) > (points + avg_candle * factor_2)) &  (points > (points.shift(-1) + avg_candle * factor_2)) ) )
def next_down_mask(points,avg_candle,factor_1 = factor_1, factor_2 = factor_2):
   return   ( ((points.shift(-1)) <= (points - (avg_candle * factor_1)))
              | ( (points > (points.shift(-1) + avg_candle * factor_2)) &  (points.shift(-1) > (points.shift(-2) + avg_candle * factor_2)) ) )
def is_down(points,avg_candle):
  #  return next_down_mask(points,avg_candle) & before_down_mask(points,avg_candle)
   return next_down_mask(points,avg_candle)
def is_steady(high_points,low_points,avg_candle):
  return ~next_up_mask(low_points,avg_candle) & ~next_down_mask(high_points,avg_candle)

def get_minor_reversals(high_points,low_points):
    avg_candle = (high_points - low_points).rolling(window=14).mean()
    down_mask = is_down(high_points,avg_candle)
    up_mask = is_up(low_points,avg_candle)
    down_reversal = high_points[((~(down_mask.shift(1) & down_mask)) & down_mask)].index
    up_reversal = low_points[((~(up_mask.shift(1) & up_mask)) & up_mask)].index
    down_reversalfl = high_points[down_mask].index
    up_reversalfl = low_points[up_mask].index
    # mask = is_steady(high_points,low_points,avg_candle)
    # steady = low_points[mask & ~mask.shift(1).fillna(True)].index
    # return up_reversal, down_reversal, steady
    return up_reversal, down_reversal, up_reversalfl, down_reversalfl


def find_outliers(chunk, q1 = 0.35, q2 = 0.65):
    Q1 = chunk.quantile(q1)
    Q3 = chunk.quantile(q2)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return chunk[(chunk < lower_bound) | (chunk > upper_bound)]
def get_trend_shift(high_points,low_points):
     avg_hl = (low_points + high_points) / 2
     avg_hl = savgol_filter(avg_hl,5,4)
     trend_angle_diff = avg_hl.diff(periods = -1).apply(lambda x: math.degrees(math.atan(x))).diff()
     chunk_size = 200
     trend_angle_diff = trend_angle_diff.reset_index(drop = True)
     group_key = trend_angle_diff.index // chunk_size
     trend_shift = trend_angle_diff.groupby(group_key).apply(find_outliers).reset_index(level=0, drop=True)
     return high_points.index[trend_shift.index]



 
def infer_time_unit(series):
    time_diffs = series.index.to_series().diff().shift(-1).dropna()
    diff_counts = Counter(time_diffs)
    most_common_diff, _ = diff_counts.most_common(1)[0]
    return most_common_diff
def calculate_unit_diffs(series, unit):
    time_diffs = series.index.to_series().diff().shift(-1).dropna()
    unit_diffs = time_diffs / unit
    return unit_diffs
def noTrueBefore(mask):
    return (~(mask.shift(1).fillna(False)) & mask)
def calculate_angle(row):
    # tan = opp / adj, in my case opp is the y, and math.atan takes (opp,adj)
    return math.degrees(math.atan2(row.y2 - row.y1,row.x_unit_scaled * row.units))

def get_angles(series, unit_series):
    price_range = unit_series.diff().abs().rolling(30).mean().bfill() * 20
    price_range = price_range[series.index]
    x_unit_scaled = price_range / 100

    time_unit = infer_time_unit(unit_series)
    nunits = calculate_unit_diffs(series,time_unit)

    angle_calc_df = pd.DataFrame({"y1":series,"y2":series.shift(-1),"units":nunits,"x_unit_scaled":x_unit_scaled}).bfill()
    angles = angle_calc_df.apply(calculate_angle,axis = 1)
    return angles

def get_CriticalPoints(angles):
    up1 = angles[noTrueBefore((angles > 30) & (angles < 60))].index
    up2 = angles[noTrueBefore((angles > 60))].index
    std = angles[noTrueBefore((angles > -30) & (angles < 30))].index
    down1 = angles[noTrueBefore((angles < -30) & (angles > -60))].index
    down2 = angles[noTrueBefore((angles < -60))].index
    return up1,up2,std,down1,down2

def get_BuySellStd(angles,limit):
    std = angles[noTrueBefore((angles > -limit) & (angles < limit))].index
    buy = angles[noTrueBefore((angles > limit))].index
    sell = angles[noTrueBefore((angles < -limit))].index
    return buy, sell, std

def get_BuySellStd_cleaned(avg_hl,angles):
    buy, sell, std = get_BuySellStd(angles,15)
    signals = pd.concat([buy.to_series(),std.to_series(),sell.to_series()]).sort_values()
    has_after = (pd.Series(angles.index,index = angles.index).shift(-1)[signals]).isin(signals)
    has_before = (pd.Series(angles.index,index = angles.index).shift(1)[signals]).isin(signals)
    signals = signals[~(has_after & has_before)]
    filtered_avg_hl = avg_hl[signals]
    filtered_avg_hl[avg_hl.index[-1]] = avg_hl[-1]
    filtered_angles = get_angles(filtered_avg_hl,avg_hl)
    buy, sell, std = get_BuySellStd(filtered_angles,15)
    return buy, sell, std

def get_major_Buy(row,buy_value):
   c_sell = row.current_sell
   p_sell = row.previous_sell
   chunk_buy = buy_value[(buy_value.index < c_sell) & (buy_value.index > p_sell)]
   if len(chunk_buy) > 0:
     return chunk_buy.idxmin()

def get_major_BuySellStd(avg_hl,sell,buy):
    sell_line = avg_hl[sell]
    angles = get_angles(sell_line,avg_hl)
    Mbuy, Msell, Mstd = get_BuySellStd(angles,10)
    Msell_series = Msell.to_series()
    df = pd.DataFrame({"current_sell":Msell_series,"previous_sell":Msell_series.shift().fillna(avg_hl.index[0])})
    buy_value = avg_hl[buy]
    Mbuy = df.apply(get_major_Buy, args=(buy_value,),axis = 1).dropna()
    Mbuy = pd.DatetimeIndex(Mbuy)
    return Mbuy, Msell, Mstd


 
from scipy.stats.mstats import winsorize

def atr(data, window=14, limits=(0.05, 0.05)):

    true_range = np.maximum(data['High'] - data['Low'],
                            np.maximum(abs(data['High'] - data['Close'].shift()),
                                       abs(data['Low'] - data['Close'].shift())))

    tr_winsorized = winsorize(true_range, limits=limits)

    atr_winsorized = pd.Series(tr_winsorized).rolling(window=window).mean()

    return atr_winsorized.fillna(method='bfill').to_numpy()

 
def is_signal_strategy_viable(data, tp_atr = 3, sl_atr = 1,search_window = 13):
    atr_values = atr(data)

    high_prices = data.High.values
    low_prices = data.Low.values
    close_prices = data.Close.values
    open_prices = data.Open.values

    long_target_prices = open_prices + (atr_values * tp_atr)
    long_stop_losses = open_prices - (atr_values * sl_atr)
    short_target_prices = open_prices - (atr_values * tp_atr)
    short_stop_losses = open_prices + (atr_values * sl_atr)

    long_viability_mask = []
    short_viability_mask = []

    for current_index in range(len(data) - 1):
        next_index = current_index + 1
        open_price = open_prices[next_index]

        long_target_price = long_target_prices[next_index]
        long_stop_loss = long_stop_losses[next_index]
        short_target_price = short_target_prices[next_index]
        short_stop_loss = short_stop_losses[next_index]

        future_lows = low_prices[next_index:next_index + search_window]
        future_highs = high_prices[next_index:next_index + search_window]

        long_target_hit_arr = np.where(future_highs > long_target_price)[0]
        short_target_hit_arr = np.where(future_lows < short_target_price)[0]

        long_viable = is_long_target_hit_arr_valid(long_target_hit_arr, future_lows, long_stop_loss)
        long_viability_mask.append(long_viable)
        if not long_viable:
          short_viability_mask.append(is_short_target_hit_arr_valid(short_target_hit_arr, future_highs, short_stop_loss))
        else:
            short_viability_mask.append(False)


    return (data.index[:-1][long_viability_mask], data.index[:-1][short_viability_mask])

def is_long_target_hit_arr_valid(target_hit_arr, future_lows, stop_loss):
    if target_hit_arr.size == 0:
        return False
    else:
        first_target_hit_index = target_hit_arr[0]
        if first_target_hit_index == 0:
          if future_lows[0] < stop_loss:
             return False
          else:
             return True
        else:
          lows_before_target_hit = future_lows[:first_target_hit_index]
          if np.any(lows_before_target_hit <= stop_loss):
              return False
          else:
              return True

# 2 functions to avoid more if statemnts
def is_short_target_hit_arr_valid(target_hit_arr, future_highs, stop_loss):
    if target_hit_arr.size == 0:
        return False
    else:
        first_target_hit_index = target_hit_arr[0]
        if first_target_hit_index == 0:
          if future_highs[0] > stop_loss:
             return False
          else:
             return True
        else:
          highs_before_target_hit = future_highs[:first_target_hit_index]
          if np.any(highs_before_target_hit >= stop_loss):
              return False
          else:
              return True



 
def get_maxs(points):
   maxs_index = []
   maxs_value = []
   if len(points) > 2:
     for i in range(1,len(points)-1):
        if points[i] > points[i-1] and points[i] > points[i+1]:
          maxs_index.append(i)
          maxs_value.append(points[i])
   return maxs_index,maxs_value

def get_mins(points,maxs):
    mins = []
    ranges = maxs
    for i in range(len(maxs)-1):
      idxmin_loc = points.iloc[ranges[i]+1:ranges[i+1]].idxmin()
      idx_loc = points.index.get_loc(idxmin_loc)
      mins.append(idx_loc)
    return mins

def get_major_reversals(high_points,low_points):
    maxs_index,maxs_value = get_maxs(high_points)
    max_maxs_index,maxs_value = get_maxs(maxs_value)
    maxs_index = [maxs_index[i] for i in max_maxs_index]
    mins_index = get_mins(low_points,maxs_index)
    return high_points.index[mins_index],high_points.index[maxs_index]

 
def get_reversals_savgol(df):
  avg_hl = (df.High + df.Low) / 2
  avg_hl_smoothed = savgol_filter(avg_hl,5,2)
  atr = ta.atr(high = df.High, low = df.Low, close = df.Close)
  maxs = find_peaks(avg_hl_smoothed, distance = 15, width = 3, prominence = atr.iloc[-1])
  mins = find_peaks(-1*avg_hl_smoothed, distance = 15, width = 3, prominence = atr.iloc[-1])
  return mins[0], maxs[0]

 
def directional_change(close, high, low, sigma):
    high = low = close
    atr = ta.atr(high,low,close).fillna(method='bfill')
    up_zig = True # Last extreme is a bottom. Next is a top.
    tmp_max = high[0]
    tmp_min = low[0]
    tmp_max_i = low.index[0]
    tmp_min_i = low.index[0]
    tops = []
    bottoms = []
    for i in range(len(close)):
        if up_zig: # Last extreme is a bottom
            if high[i] > tmp_max:
                tmp_max = high[i]
                tmp_max_i = low.index[i]
            elif close[i] < tmp_max - atr[i] * sigma:
                tops.append(tmp_max_i)
                up_zig = False
                tmp_min = low[i]
                tmp_min_i = low.index[i]
        else:
            if low[i] < tmp_min:
                tmp_min = low[i]
                tmp_min_i = low.index[i]
            elif close[i] > tmp_min + atr[i] * sigma:
                bottoms.append(tmp_min_i)
                up_zig = True
                tmp_max = high[i]
                tmp_max_i = low.index[i]

    return tops, bottoms



 
# Major_Trend (categorical): -1,0,1
def get_major_trend(mins_index,maxs_index,original_idx_list,len_df):
    idx_list = np.insert(original_idx_list, 0, -1, axis=0)
    idx_list = np.append(idx_list, len_df)
    trend_array = np.zeros(len_df)
    trend = -1 if min(mins_index) < min(maxs_index) else 1
    for i in range(len(idx_list)-1):
        trend_array[idx_list[i]+1:idx_list[i+1]] = trend
        trend *= -1
    return trend_array

# Number of candles before next major reversal (discrete): 0 to inf
# Change before next major reversal (continuous): 0 to inf
def get_Metrics_Before_Next_Major_Reversal(points,original_idx_list,len_df):
    change_array = np.array([], dtype=int)
    nCandles_array = np.array([], dtype=int)
    prev_reversal = 0
    for reversal in original_idx_list:
        num_points = reversal - prev_reversal
        nCandles_array = np.concatenate((nCandles_array, np.arange(num_points, 0, -1)))
        partition = points.iloc[prev_reversal:reversal]
        change_array = np.concatenate((change_array, ((partition * -1) + points.iloc[reversal]) / partition))
        prev_reversal = reversal
    nCandles_array = np.concatenate((nCandles_array, np.full(len_df - reversal, np.nan)))
    change_array = np.concatenate((change_array, np.full(len_df - reversal, np.nan)))
    return nCandles_array, change_array

 
def get_potential_profit(buy, sell, std, Open, fee = 0.1):
    signals = pd.concat([buy.to_series(),std.to_series(),sell.to_series()]).sort_values()
    entries = Open.shift(-1)[signals]
    exits = Open.shift(-1)[signals.shift(-1).ffill()]
    fees = (entries * 0.001).values + (exits * 0.001).values
    potential_profit = (exits.values - entries.values)
    potential_profit = pd.Series(potential_profit, index = signals)
    potential_profit[sell] = potential_profit[sell] * -1
    potential_profit = (potential_profit - fees) / entries.values * 100
    potential_profit.drop(std,inplace = True)
    return potential_profit[:-1]


 
def label_angles(df):
  avg_hl = (df.High + df.Low) / 2
  angles = get_angles(avg_hl,avg_hl)
  buy, sell, std = get_BuySellStd_cleaned(avg_hl,angles)
  Mbuy, Msell, Mstd = get_major_BuySellStd(avg_hl,sell,buy)
  df["Reversal"] = 0
  df["Reversal"][buy] = 1
  df["Reversal"][sell] = 2
  df["Reversal"][std] = 3
  # df["Reversal"][Mbuy] = 4
  # df["Reversal"][Msell] = 5
  # df["Reversal"][Mstd] = 6
  buy_sell = pd.concat([buy.to_series(),sell.to_series()]).sort_values()
  Mbuy_Msell = pd.concat([Mbuy.to_series(),Msell.to_series()]).sort_values()
  df["Potential_profit"] = 0
  # df["Potential_profit"][buy_sell[:-1].index] = get_potential_profit(buy, sell, std , df.Open)
  df["Potential_profit"][Mbuy_Msell[:-1].index] = get_potential_profit(Mbuy, Msell, Mstd , df.Open)
  return df

def label_signal_strategy_viable(df):
  df["Reversal"] = 0
  long_entry,short_entry = is_signal_strategy_viable(df)
  df["Reversal"][long_entry] = 1
  df["Reversal"][short_entry] = 2
  return df

def label_major_reversals(df):
   mins, maxs = get_major_reversals(df.High, df.Low)
   df["Reversal"] = 0
   df["Reversal"][mins] = 1
   df["Reversal"][maxs] = 2
   return df


 
# df_labeled = label_angles(df)
# df_labeled = label_major_reversals(df)
df_labeled = label_signal_strategy_viable(df)

 
def plot_pivot(num_plots,plot_size,df):
  if num_plots*plot_size > len(df):
    print("There isn`t enough data")
    return
  list_df = [df[i:i+plot_size] for i in range(0,num_plots*plot_size,plot_size)]
  for chunk in list_df:
    fig = go.Figure(data=[go.Candlestick(x=chunk.index,
                    open=chunk['Open'],
                    high=chunk['High'],
                    low=chunk['Low'],
                    close=chunk['Close'])])

    for i in df["Reversal"].unique():
       if i != 0:
         chunk_reversal = df.index[(df["Reversal"] == i) & (df.index.isin(chunk.index))]
         fig.add_trace(go.Scatter(x=chunk_reversal, y=((chunk.Low + chunk.High) / 2)[chunk_reversal].sort_index(), text=str(i), mode='markers+text', marker=dict(size=5), textposition='top center'))

    # pp = df["Potential_profit"][(df["Potential_profit"].index >= chunk.index[0]) & (df["Potential_profit"].index <= chunk.index[-1]) & (df["Reversal"] != 0)]
    # fig.add_trace(go.Scatter(x=chunk_reversal, y=chunk.Low[chunk_reversal].sort_index() - chunk.Low.mean()*0.001, mode='text',text = pp.round(2),textfont=dict(size=10)) )
    fig.show()

plot_pivot(10,200,df_labeled)

 
df_labeled.to_csv("drive/MyDrive/Colab Notebooks/Trading Bot/Data/Crypto/SOL/SOLUSDT-5m-2022-2023-labeled")


