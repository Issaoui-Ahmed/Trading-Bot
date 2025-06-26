import pandas as pd
from lightweight_charts import Chart


if __name__ == '__main__':
    
    chart = Chart()
    
    # Columns: time | open | high | low | close | volume 
    df = pd.read_csv('Data/Raw/SOLUSDT-5m-2022-2023.csv')
    chart.set(df[:1000])
    
    chart.marker_list([{"time": f"{df.Date[998]}", "position": "below", "shape": "arrow_up", "color": "#2196F3", "text": ""}])
    
    chart.show(block=True)
