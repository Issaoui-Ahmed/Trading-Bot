import pandas as pd
import numpy as np


def generate_tuples(actions,df,sl = None, tp = None):

      actions = actions.shift(1).fillna(0).copy()

      timeframe = df.index.to_series().diff().min()

      transitions = { "None": [1, 3, 5, 6],
                      "Long": [2, 5],
                      "Short": [4, 6]}

      position = "None"
      long_positions = []
      short_positions = []

      action_to_position = {1: "Long", 2: "None", 3: "Short", 4: "None",5: "Short", 6: "Long"}
      action_to_append = {
          2: ("long_positions", "None"),
          4: ("short_positions", "None"),
          5: ("long_positions", "Short"),
          6: ("short_positions", "Long")
      }

      mltp_dict = {"Long": 1, "Short": -1}
      positions_dict = {"Long": long_positions,"Short": short_positions}
      sl_ref_dict = {"Long": df.Low , "Short": df.High}
      tp_ref_dict = {"Long": df.High , "Short": df.Low}
      if sl:
        sp_dict = {"Long": (1 - (sl / 100)) , "Short": (1 + (sl / 100))}
      if tp:
        tp_dict = {"Long": (1 + (tp / 100)) , "Short": (1 - (tp / 100))}
      
      for i, action in enumerate(actions):
          possible_actions = transitions[position]
          if action in possible_actions:
              current_index = df.index[i]
              if position == "None":
                 position = action_to_position[action]
                 entry_index = current_index
              else:
                 append_list_name, position = action_to_append[action]
                 position = position
                 eval(append_list_name).append((entry_index, current_index))
                 if action in [5,6]:
                    entry_index = current_index
          
          if sl and tp:
            if position != "None":
                current_index = df.index[i]
                entry_price = df.Open[entry_index]
                open_price = df.Open.iloc[i]
                mltp = mltp_dict[position]
                position_tuple = positions_dict[position]
                if sl:
                  sl_ref = sl_ref_dict[position]
                  sl_price = entry_price * sp_dict[position]
                  if mltp * open_price <= mltp * sl_price:
                    position_tuple.append((entry_index, current_index))
                    position = "None"
                  elif mltp * sl_ref.iloc[i] <= mltp * sl_price:
                    if entry_index == current_index:
                        duration = timeframe / 3
                        position_tuple.append((entry_index, entry_index + duration,sl_price))
                    else:
                        position_tuple.append((entry_index, current_index,sl_price))
                    position = "None"
                if tp:
                  if position != "None":
                    tp_price = entry_price * tp_dict[position]
                    tp_ref = tp_ref_dict[position]
                    if mltp * open_price >= mltp * tp_price:
                        position_tuple.append((entry_index, current_index))
                        position = "None"
                    elif mltp * tp_ref.iloc[i] >= mltp * tp_price:
                        if entry_index == current_index:
                          duration = timeframe / 3
                          position_tuple.append((entry_index, entry_index + duration,tp_price))
                        else:
                          position_tuple.append((entry_index, current_index,tp_price))
                        position = "None"
      return long_positions, short_positions