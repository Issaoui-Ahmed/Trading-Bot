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
    fees = fees / 100
    fees = entry_price * fees + exit_price * fees
    m = 1 if trade_type == "long" else -1
    return ((exit_price - entry_price) * m - fees) / entry_price * 100

def get_trade_duration(position):
    return (position[1] - position[0]).total_seconds()

def create_trade_df(long_positions, short_positions, price, fees):
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


def bootstrap_mean(data, n_bootstraps=1000, conf_level=0.95):
    bootstrap_means = np.array([
        data.sample(frac=1, replace=True).mean() for _ in range(n_bootstraps)
    ])
    
    lower_percentile = 100 * (1 - conf_level) / 2
    upper_percentile = 100 * (1 - (1 - conf_level) / 2)
    confidence_interval = np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    return bootstrap_means.mean(), confidence_interval


def stat_analyis(observations):
    b_mean, b_mean_interval = bootstrap_mean(observations)
    return [observations.min(), observations.max(),observations.mean(), b_mean, b_mean_interval]

def plot_kde(observations):
    mean_value = observations.mean()
    plt.figure(figsize=(5, 3))
    sns.kdeplot(observations, bw_adjust=0.5, fill=True)
    plt.axvline(mean_value, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    plt.legend()
    plt.show()

def get_duration_unit(trades):
  durations_units = {86400:"days", 3600:"hours", 60: "minutes", 1: "seconds"}
  avg_trade_duration = trades["duration"].mean()
  for duration in durations_units:
      if (avg_trade_duration / duration) > 1:
          break
  return duration, durations_units[duration]

def eval_dfs(trades):
    duration, unit = get_duration_unit(trades)
    stat_df = pd.DataFrame(columns = ["Variable","Min","Max","Mean","B_mean","B_mean_interval"])
    variables = {"Return per Trade": trades["return"],
                 "Return per Day": trades["return"].resample('D').sum(),
                 "Exp Return per Day":((trades["return"] / 100 + 1).resample('D').prod() - 1) * 100,
                 "Return per Week": trades["return"].resample('W').sum(),
                 "Exp Return per Week":((trades["return"] / 100 + 1).resample('W').prod() - 1) * 100,
                 "Return per Month": trades["return"].resample('M').sum(),
                 "Exp Return per Month":((trades["return"] / 100 + 1).resample('M').prod() - 1) * 100,
                 f"Trade Duration in {unit}": trades["duration"] / duration,
                 "Freq per Day": trades["return"].resample('D').count(),
                 "Freq per Week": trades["return"].resample('W').count(),
                 "Freq per Month": trades["return"].resample('M').count()}
    
    for variable in variables:
        stats = stat_analyis(variables[variable])
        stat_df.loc[len(stat_df)] = [variable] + stats
    
    global_df = pd.DataFrame(columns = ["Backtest From/To","Winrate", "Total Num Trades"])
    num_winning_trades = len(trades[trades["return"] > 0])
    win_rate = num_winning_trades / len(trades)
    global_df.loc[len(global_df)] = [f"{trades.index[0]} to {trades.index[-1]}", win_rate, len(trades)]
    return stat_df, global_df
    


def eval_strategy(long_positions, short_positions, price, fees = 0.1):
  trades = create_trade_df(long_positions, short_positions, price, fees)
  stat_df, global_df = eval_dfs(trades)
#   global_df = global_df.style.format("{:.3f}")
#   stat_df = stat_df.style.format("{:.3f}")
  print(global_df)
  print(stat_df)
  
   