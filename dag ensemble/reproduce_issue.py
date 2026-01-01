
import sys
import os
import pandas as pd
import numpy as np

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'backend')))

from target_engineering.meta_model_targets import engineer_targets

def check_distribution():
    file_path = r'c:\Users\aissa\OneDrive\Desktop\trading\dag ensemble\backend\datasets\SOL_USD_1m.csv'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows.")
    
    # Ensure column names are lower case
    df.columns = [c.lower() for c in df.columns]
    
    if 'close' not in df.columns:
        print("Column 'close' not found.")
        print(df.columns)
        return

    df_targets = engineer_targets(df)
    
    if 'Target_Meta' not in df_targets.columns:
        print("Target_Meta not created.")
        return
        
    print("\nTarget_Meta value counts:")
    print(df_targets['Target_Meta'].value_counts(normalize=True))
    print("\nRaw counts:")
    print(df_targets['Target_Meta'].value_counts())

if __name__ == "__main__":
    check_distribution()
