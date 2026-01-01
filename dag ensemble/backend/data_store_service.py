import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, List

class DataStoreService:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        # Cache key: filename (e.g., "data_storage.parquet") -> DataFrame
        self.cache: Dict[str, pd.DataFrame] = {}
        
        # Ensure directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Ensure default file exists
        self._ensure_default_file()

    def _ensure_default_file(self):
        default_path = os.path.join(self.storage_dir, "data_storage.parquet")
        if not os.path.exists(default_path):
            try:
                pd.DataFrame().to_parquet(default_path)
            except Exception as e:
                print(f"DataStoreService: Failed to create default file: {e}")

    def _get_file_path(self, filename: str) -> str:
        # Security: basic sanitization to prevent directory traversal
        filename = os.path.basename(filename)
        if not filename.endswith(".parquet"):
            filename += ".parquet"
        return os.path.join(self.storage_dir, filename)

    def list_files(self) -> List[str]:
        """Returns list of available parquet files."""
        files = []
        if os.path.exists(self.storage_dir):
            for f in os.listdir(self.storage_dir):
                if f.endswith(".parquet"):
                    files.append(f)
        return files

    def load(self, filename: str = "data_storage.parquet") -> pd.DataFrame:
        """
        Loads data from disk or cache.
        """
        if not filename:
            filename = "data_storage.parquet"
            
        # 1. Check Cache
        if filename in self.cache:
            return self.cache[filename]
        
        # 2. Check Disk
        file_path = self._get_file_path(filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                if not df.index.is_monotonic_increasing:
                    df.sort_index(inplace=True)
                self.cache[filename] = df
                return df
            except Exception as e:
                print(f"DataStoreService: Error loading {filename}: {e}")
                return pd.DataFrame()
        
        # 3. Return Empty
        return pd.DataFrame()

    def update(self, new_data: pd.DataFrame, filename: str = "data_storage.parquet") -> pd.DataFrame:
        """
        Updates the stored data in the specified file with new_data using Upsert.
        """
        if not filename:
            filename = "data_storage.parquet"
            
        if new_data is None or new_data.empty:
            return self.load(filename)

        current_df = self.load(filename)
        
        if current_df.empty:
            updated_df = new_data.copy()
        else:
            # 1. Coordinate Columns
            all_cols = list(set(current_df.columns).union(set(new_data.columns)))
            current_aligned = current_df.reindex(columns=all_cols)
            new_aligned = new_data.reindex(columns=all_cols)
            
            # 2. Upsert: New data overwrites old data at intersection
            updated_df = new_aligned.combine_first(current_aligned)
            
        updated_df = updated_df.convert_dtypes()
        
        try:
            if updated_df.index.name is None:
                updated_df.index.name = 'index'
            
            # [FIX] Ensure Chronological Order for Eval Node
            if not updated_df.index.is_monotonic_increasing:
                updated_df.sort_index(inplace=True)
            
            # [FIX] Enforce Index Type if new data was Datetime and current was not (or empty)
            if not updated_df.empty and isinstance(new_data.index, pd.DatetimeIndex):
                 if not isinstance(updated_df.index, pd.DatetimeIndex):
                      # Attempt force convert
                      updated_df.index = pd.to_datetime(updated_df.index)
                 updated_df.sort_index(inplace=True)
                
            updated_df.to_parquet(self._get_file_path(filename))
            self.cache[filename] = updated_df
            
        except Exception as e:
            print(f"DataStoreService: Save failed for {filename}: {e}")
            
        return updated_df

    def clear_all(self):
        """Clears all managed data stores (cache only or files too? User said clear on reset).
           Let's clear cache and maybe empty the files but keep them?
           Actually, usually Reset means 'start fresh'. 
           For persistent storage, maybe we SHOULD NOT clear on workflow reset?
           The User said: "Data Store clears its stored data upon workflow reset." in previous context.
           But now it's "Write & Read" persistent storage.
           I will stick to clearing the *content* of the files but keeping the files.
        """
        self.cache.clear()
        
        # Clear content of all parquet files in directory
        if os.path.exists(self.storage_dir):
            for f in os.listdir(self.storage_dir):
                if f.endswith(".parquet"):
                    try:
                        path = os.path.join(self.storage_dir, f)
                        # Overwrite with empty DF
                        pd.DataFrame().to_parquet(path)
                    except Exception as e:
                        print(f"Error clearing {f}: {e}")
