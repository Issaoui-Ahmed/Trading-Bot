import plotly.graph_objects as go
from plotly.offline import plot
from lightweight_charts import Chart
import pandas as pd


# error analysis:

# - plot the true use tve function
# - plot predicted
# - plot positions

def plot_signals(num_plots,plot_size,df,signals,long_positions, short_positions):
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

    for i in signals.unique():
       if i != 0:
         chunk_reversal = df.index[(signals == i) & (df.index.isin(chunk.index))]
         fig.add_trace(go.Scatter(x=chunk_reversal, y=((chunk.Low + chunk.High) / 2)[chunk_reversal].sort_index(), text=str(int(i)), mode='markers+text', marker=dict(size=3), textposition='top center'))
         
    for positions, color in zip([long_positions, short_positions], ['green', 'red']):
            for start, end in positions:
                if start in chunk.index and end in chunk.index:
                    fig.add_trace(go.Scatter(x=[start, end],
                                             y=[chunk.at[start, 'Close'], chunk.at[end, 'Close']],
                                             mode='markers+lines',
                                             marker=dict(color=color, size=8),
                                             name=f'{"Long" if color == "green" else "Short"} Position'))
    # fig.show(render = "vscode")
    plot(fig, filename='my_plot.html', auto_open=True)
    

import pandas as pd
from lightweight_charts import Chart


def tv_plot(df):    
    chart = Chart()
    chart.set(df[:1000])
    
    chart.marker_list([{"time": f"{df.Date[998]}", "position": "below", "shape": "arrow_up", "color": "#2196F3", "text": ""}])
    
    chart.show(block=True)