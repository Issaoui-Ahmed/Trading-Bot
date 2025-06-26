import pandas as pd
import pandas_ta as ta

import ML_usage as mlu
import Task_Evaluation as te
import Feature_Eval as fev
import Error_Analysis as ea

from importlib import reload
def reload_modules():
    for module in [fev, te, mlu, ea]:
        reload(module)   
        
from lightweight_charts import Chart
      

# load data and prep
def load_data(path):
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["Date"])
    df.drop(columns = ["Date"],inplace = True)
    df.sort_index(inplace = True)
    df.index = pd.to_datetime(df.index)
    return df

sol_path = "../Data/Raw/SOLUSDT-5m-2022-2023.csv"
qgm_path = "../Data/Raw/QGMD_EOD.csv"
df = load_data(qgm_path)
# get indicators
super_trend = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3)
super_trend_direction = super_trend.iloc[:,1].copy()

st_long_mask = (super_trend_direction == 1)
dema_long_mask = df.Close > ta.dema(df.Close, length = 10)
st_close_long_mask = (super_trend_direction == -1)

long_mask = st_long_mask 
close_long_mask = st_close_long_mask
signals = pd.Series(0, index = df.index)
signals[long_mask] = 1
signals[close_long_mask] = 2

# make strategy
long_positions, short_positions = mlu.generate_tuples(pd.Series(signals), df, tp = None, sl = None) 


# see strategy (indicators , signals and positions)
reload_modules()
# ea.plot_signals(2,200,df,signals,long_positions,short_positions)

# tev
# te.eval_strategy(long_positions, short_positions, df.Open,fees = 0)

def create_markers_list():
    pass

# use in notebook: https://github.com/TechfaneTechnologies/pytvlwcharts?tab=readme-ov-file
# watch all https://www.youtube.com/watch?v=TlhDI3PforA&t=947s

if __name__ == '__main__':
    chart = Chart()
    chart.set(df[:1000])
    
    chart.marker_list([{"time": f"{df.index[998]}", "position": "below", "shape": "arrow_up", "color": "#2196F3", "text": ""}])
    
    chart.show(block=True)
