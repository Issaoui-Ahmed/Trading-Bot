import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

def get_trade_return(position, trade_type,price,fees):
    entry_price = price[position[0]]
    if len(position) == 3:
        exit_price = position[2]
    else:
      exit_price = price[position[1]]
    fees = entry_price * fees + exit_price * fees
    m = 1 if trade_type == "long" else -1
    return ((exit_price - entry_price) * m - fees) / entry_price * 100

def get_trade_duration(position):
    return (position[1] - position[0]).seconds / 60

def create_trade_df(long_positions, short_positions, price, fees = 0.001):
    trade_data = []

    def prepare_trades(positions, trade_type):
        for position in positions:
            trade_return = get_trade_return(position, trade_type,price,fees)
            duration = get_trade_duration(position)
            trade_data.append([position[0], trade_return, duration, trade_type])

    prepare_trades(long_positions, 'long')
    prepare_trades(short_positions, 'short')

    trades = pd.DataFrame(trade_data, columns=['entry_date', 'return', 'duration', 'type'])

    trades.set_index('entry_date', inplace=True)
    trades.sort_index(inplace=True)

    return trades

def value_estimation(observations, variable_name):

    print(f"\nEstimation for: {variable_name}")
    lower_bound = observations.quantile(0.025)
    upper_bound = observations.quantile(0.975)
    print(f"The 95% confidence interval of the observations is approximately from {lower_bound:.2f} to {upper_bound:.2f}.\n")
    mean_value = observations.mean()
    plt.figure(figsize=(5, 3))
    sns.kdeplot(observations, bw_adjust=0.5, fill=True)
    plt.axvline(mean_value, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.legend()
    plt.show()

def return_estimations(trades_df):
  value_estimation(trades_df["return"],"Return per trade")
  value_estimation(trades_df["return"].resample('D').sum(),"Return per day")
  value_estimation(((trades_df["return"] / 100 + 1).resample('D').prod() - 1) * 100,"Exp return per day")

  value_estimation(trades_df["return"].resample('W').sum(),"Return per week")
  value_estimation(((trades_df["return"] / 100 + 1).resample('W').prod() - 1) * 100,"Exp return per week")

  value_estimation(trades_df["return"].resample('M').sum(),"Return per month")
  value_estimation(((trades_df["return"] / 100 + 1).resample('M').prod() - 1) * 100,"Exp return per month")

def exc(long_positions, short_positions, price, fees = 0.001):
   trades = create_trade_df(long_positions, short_positions, price)
   return_estimations(trades)