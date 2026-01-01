import pandas as pd
import os
import time

class DataReplayer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = pd.DataFrame()
        self.current_index = 0
        self.loaded = False
        
    def load_dataset(self):
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found: {self.dataset_path}")
            return False
            
        try:
            self.df = pd.read_csv(self.dataset_path)
            
            # Standardize Timestamp Index if column exists
            if 'timestamp' in self.df.columns:
                # Convert to datetime (assuming unix seconds if numeric, or ISO string)
                try:
                    # Check if numeric
                    if pd.api.types.is_numeric_dtype(self.df['timestamp']):
                        # Simple heuristic: If max value > 3e10 (Year 2920), assume MS or NS
                        # If max value around 1.7e9, it's seconds.
                        max_ts = self.df['timestamp'].max()
                        unit = 's'
                        if max_ts > 3e11: # > Year 11000 roughly, likely ms (1.7e12) or ns
                             unit = 'ms'
                        
                        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], unit=unit)
                    else:
                        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                    
                    self.df.set_index('timestamp', inplace=True)
                    self.df.sort_index(inplace=True)
                except Exception as e:
                    print(f"Error parsing timestamp in {self.dataset_path}: {e}")

            self.loaded = True
            self.current_index = 0
            return True
        except Exception as e:
            print(f"Error loading dataset {self.dataset_path}: {e}")
            return False

    def next_row(self):
        if not self.loaded or self.current_index >= len(self.df):
            return None
            
        # Get row as DataFrame (single row)
        # Allows preserving the index (Timestamp)
        row = self.df.iloc[[self.current_index]].copy()
        self.current_index += 1
        
        return row
        
    def has_more(self):
        return self.loaded and self.current_index < len(self.df)
