import os
import sys
import time
import asyncio
import threading
import json
import random
import pickle
import numpy as np
import pandas as pd
import importlib.util
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

def convert_numpy(obj):
    """
    Recursively converts numpy types to native Python types.
    Handles NaN/Inf by converting to None.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.ndarray):
        return convert_numpy(obj.tolist())
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, set):
        return [convert_numpy(v) for v in list(obj)]
    elif isinstance(obj, pd.Series):
        return convert_numpy(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return convert_numpy(obj.to_dict('records'))
    elif isinstance(obj, pd.Index):
        return convert_numpy(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    # Catch-all for other nullable types (pd.NA)
    if pd.isna(obj):
        return None
    return obj

# Integration of Model Registry
from model_registry import ModelRegistry
from data_replayer import DataReplayer
from data_store_service import DataStoreService

@asynccontextmanager
async def lifespan(app: FastAPI):
    global REGISTRY
    print("Initializing Model Registry...")
    REGISTRY = ModelRegistry(PRETRAINED_MODELS_DIR)
    yield
    # Clean up resources if needed

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_MODELS_DIR = os.path.join(BASE_DIR, "pretrained_models")
FEATURE_SCRIPTS_DIR = os.path.join(BASE_DIR, "feature_engineering")
REGISTRY = None  # Initialized in startup

LATEST_DATA = {}
FULL_NODE_OUTPUTS = {} # [NEW] Stores full outputs for "Expand" feature
LATEST_INFERENCE_RESULTS = {}
WORKFLOW_CONFIG = {}
IS_RUNNING = False
IS_PAUSED = False # [NEW] Global Pause State
REPLAY_INSTANCES = {} # node_id -> DataReplayer Instance
REPLAY_STATS = {} # node_id -> {current, total, progress}

# Data Store State
# node_id -> pd.DataFrame (Accumulated History)
# DATA_STORE_STATES = {} # Deprecated in favor of DataStoreService
DATA_STORE_STORAGE_DIR = os.path.join(BASE_DIR, "data_store_storage")
DATA_STORE_SERVICE = DataStoreService(DATA_STORE_STORAGE_DIR)

# Paper Trading State
# node_id -> { "cash": float, "holdings": float, "history": [], "initial_capital": float }
PAPER_WALLETS = {}

# Ensure feature directory exists
os.makedirs(FEATURE_SCRIPTS_DIR, exist_ok=True)

# Ensure saved workflows directory exists
SAVED_WORKFLOWS_DIR = os.path.join(BASE_DIR, "saved_workflows")
os.makedirs(SAVED_WORKFLOWS_DIR, exist_ok=True)



@app.get("/feature-scripts")
def get_feature_scripts():
    """Returns list of available python feature engineering scripts."""
    scripts = []
    if os.path.exists(FEATURE_SCRIPTS_DIR):
        for f in os.listdir(FEATURE_SCRIPTS_DIR):
            if f.endswith(".py") and f != "__init__.py":
                scripts.append(f)
    return {"scripts": scripts}

TRADING_BRAINS_DIR = os.path.join(BASE_DIR, "trading_brains")
os.makedirs(TRADING_BRAINS_DIR, exist_ok=True)

DATASETS_DIR = os.path.join(BASE_DIR, "datasets") # [RENAME]
os.makedirs(DATASETS_DIR, exist_ok=True)

@app.get("/datasets")
def get_datasets():
    """Returns list of available CSV datasets."""
    files = []
    if os.path.exists(DATASETS_DIR):
        for f in os.listdir(DATASETS_DIR):
            if f.endswith(".csv"):
                files.append(f)
    return {"datasets": files}

@app.get("/data-storage-files")
def get_data_storage_files():
    """Returns list of available parquet files for Write & Fetch nodes."""
    files = DATA_STORE_SERVICE.list_files()
    # Ensure default exists in list if not created yet (though service init does it)
    if "data_storage.parquet" not in files:
        files.append("data_storage.parquet")
    return {"files": files}

@app.get("/storage/content")
def get_storage_content(filename: str = "data_storage.parquet", rows: int = 100):
    """Returns the content of a specific storage file."""
    df = DATA_STORE_SERVICE.load(filename)
    if df.empty:
        return {"data": []}
    
    # Return last N rows
    subset = df.tail(rows).copy()
    
    # Ensure timestamp format
    if 'timestamp' not in subset.columns and subset.index.name == 'timestamp':
         subset.reset_index(inplace=True)
         
    if 'timestamp' not in subset.columns:
        subset['timestamp'] = subset.index.astype(str)
        
    return {"data": convert_numpy(subset.to_dict('records'))}

@app.post("/workflow/pause")
def pause_workflow():
    global IS_PAUSED
    IS_PAUSED = True
    return {"status": "paused", "is_paused": True}

@app.post("/workflow/resume")
def resume_workflow():
    global IS_PAUSED
    IS_PAUSED = False
    return {"status": "resumed", "is_paused": False}

@app.get("/nodes/{node_id}/output")
def get_node_full_output(node_id: str):
    """Returns the full output data for a specific node."""
    val = FULL_NODE_OUTPUTS.get(node_id)
    if val is None:
        return {"data": []}
    
    # If it's a DataFrame, convert to records
    if isinstance(val, pd.DataFrame):
        # Ensure timestamp is string for JSON
        df_copy = val.copy()
        if 'timestamp' not in df_copy.columns:
            df_copy['timestamp'] = df_copy.index.astype(str)
        return {"data": df_copy.to_dict('records')}
    
    # If dict with data frame
    if isinstance(val, dict) and 'data' in val and isinstance(val['data'], pd.DataFrame):
         df_copy = val['data'].copy()
         if 'timestamp' not in df_copy.columns:
            df_copy['timestamp'] = df_copy.index.astype(str)
         # Return the data key's content
         return {"data": df_copy.to_dict('records')}
         
    return {"data": val}

@app.post("/workflow/reset")
def reset_workflow():
    global IS_PAUSED, REPLAY_INSTANCES, REPLAY_STATS, DATA_STORE_STATES, PAPER_WALLETS, FEED_MANAGER
    
    # 1. Pause immediately to stop loop interference during reset
    IS_PAUSED = True
    
    # 2. Clear Replayers
    REPLAY_INSTANCES.clear()
    REPLAY_STATS.clear()
    
    # 3. Clear Data Store
    DATA_STORE_SERVICE.clear_all()
    # DATA_STORE_STATES.clear()
    
    # 4. Clear Paper Wallets
    PAPER_WALLETS.clear()
    
    # 5. Clear Feed Manager Buffers (Reset simulated history)
    # We should probably keep 'real' live data buffers if we were mixing, 
    # but for safety in this context, let's clear buffers that might be polluted by replay.
    # For now, we'll clear EVERYTHING to be safe. 
    FEED_MANAGER.raw_buffers.clear()
    FEED_MANAGER.unified_dataset = pd.DataFrame()
    
    # 6. Reset global LATEST data containers
    global LATEST_DATA, LATEST_INFERENCE_RESULTS, FULL_NODE_OUTPUTS
    LATEST_DATA.clear()
    FULL_NODE_OUTPUTS.clear()
    LATEST_INFERENCE_RESULTS.clear()

    print("Workflow Deep Reset Complete.")
    return {"status": "reset_complete"}

@app.get("/workflow/status")
def get_workflow_status():
    return {"is_paused": IS_PAUSED}





@app.get("/trading-brains")
def get_trading_brains():
    """Returns list of available trading brain scripts."""
    scripts = []
    if os.path.exists(TRADING_BRAINS_DIR):
        for f in os.listdir(TRADING_BRAINS_DIR):
            if f.endswith(".py") and f != "__init__.py":
                scripts.append(f)
    return {"scripts": scripts}

@app.get("/training-scripts")
def get_training_scripts():
    """Returns list of available python training scripts."""
    training_dir = os.path.join(BASE_DIR, "training")
    scripts = []
    if os.path.exists(training_dir):
        for f in os.listdir(training_dir):
            if f.endswith(".py") and f != "__init__.py":
                scripts.append(f)
    return {"scripts": scripts}

@app.get("/models")
def get_models():
    """Returns list of available models in the registry."""
    if REGISTRY is None:
        return {"models": []}
    return {"models": REGISTRY.get_model_names()}

@app.get("/models/{model_name}/metadata")
def get_model_metadata(model_name: str):
    """Returns metadata for a specific model."""
    if REGISTRY is None:
        return {}
    return REGISTRY.get_model_metadata(model_name)

# --- Feed Manager & Smart Scheduler ---
from kraken_feed import KrakenFeed
from datetime import datetime, timedelta

KRAKEN_FEED = KrakenFeed()
# Track last fetch time per node to avoid spamming: {node_id: last_fetch_timestamp}
NODE_LAST_FETCH = {}

class FeedManager:
    def __init__(self):
        # Stores raw dataframes for each (pair, timeframe) pair
        # Key: "XBTUSD_1m" -> DataFrame
        self.raw_buffers: Dict[str, pd.DataFrame] = {}
        
        # The unified dataset (merged, reindexed to smallest timeframe)
        self.unified_dataset: pd.DataFrame = pd.DataFrame()
        
        # Metadata about the current unified setup
        self.primary_timeframe = None
    
    def set_buffer(self, pair, timeframe, new_data: pd.DataFrame):
        """Replaces the buffer entirely (Stateless / Windowed mode)."""
        key = f"{pair}_{timeframe}"
        if new_data is None or new_data.empty:
            return
        self.raw_buffers[key] = new_data

    def update_buffer(self, pair, timeframe, new_data: pd.DataFrame):
        key = f"{pair}_{timeframe}"
        if new_data is None or new_data.empty:
            return

        if key not in self.raw_buffers:
            self.raw_buffers[key] = new_data
        else:
            current_df = self.raw_buffers[key]
            updated_df = pd.concat([current_df, new_data])
            updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
            updated_df.sort_index(inplace=True)
            self.raw_buffers[key] = updated_df

    def rebuild_unified_dataset(self, active_contexts):
        """
        Rebuilds the unified dataset for visualization.
        For now, simply duplicates the PRIMARY dataset (closest to 1m) without prefixes.
        This ensures /dataset endpoint works for the main flow.
        """
        if not active_contexts:
            self.unified_dataset = pd.DataFrame()
            return

        # Pick primary context
        tf_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
        
        sorted_contexts = sorted(active_contexts, key=lambda x: tf_map.get(x[1], 999999))
        primary_pair, primary_tf = sorted_contexts[0]
        self.primary_timeframe = primary_tf
        
        primary_key = f"{primary_pair}_{primary_tf}"
        if primary_key not in self.raw_buffers:
            self.unified_dataset = pd.DataFrame()
            return 
            
        # DIRECTLY use the raw buffer (Standard Columns)
        self.unified_dataset = self.raw_buffers[primary_key].copy()
        
    def get_snapshot(self, rows=5):
        if self.unified_dataset.empty:
            return []
        subset = self.unified_dataset.tail(rows).copy()
        subset['timestamp'] = subset.index.astype(str)
        return subset.to_dict('records')


class WorkflowUpdate(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

@app.post("/workflow")
def update_workflow(workflow: WorkflowUpdate):
    global WORKFLOW_CONFIG
    
    # Validation: Enforce Single Feed Context per Live Feed Node
    nodes = workflow.nodes
    edges = workflow.edges
    
    # 1. Build Adjacency List for traversal
    # 2. Validate Many-to-One restrictions
    node_map = {n['id']: n for n in nodes}
    incoming_count = {n['id']: 0 for n in nodes}
    
    for edge in edges:
        tgt = edge['target']
        if tgt in incoming_count:
            incoming_count[tgt] += 1
            
    for nid, count in incoming_count.items():
        if count > 1:
            node_type = node_map.get(nid, {}).get('type')
            if node_type != 'mergeNode':
                raise HTTPException(status_code=400, detail=f"Node {nid} has multiple inputs. Only Merge Nodes support many-to-one connections.")

    # Logic Simplified: Just update the config. All processing happens in run_inference_dag.
    WORKFLOW_CONFIG = workflow.model_dump()
    print("Workflow updated")
    return {"status": "updated"}

@app.get("/workflows")
def list_workflows():
    """Returns a list of saved workflow names."""
    workflows = []
    if os.path.exists(SAVED_WORKFLOWS_DIR):
        for f in os.listdir(SAVED_WORKFLOWS_DIR):
            if f.endswith(".json"):
                workflows.append(f.replace(".json", ""))
    return {"workflows": workflows}

@app.post("/workflows/{name}")
def save_workflow(name: str, workflow: WorkflowUpdate):
    """Saves the current workflow configuration to a file."""
    file_path = os.path.join(SAVED_WORKFLOWS_DIR, f"{name}.json")
    with open(file_path, "w") as f:
        json.dump(workflow.model_dump(), f, indent=4)
    return {"status": "saved", "name": name}

@app.get("/workflows/{name}")
def load_workflow(name: str):
    """Loads a workflow configuration from a file."""
    file_path = os.path.join(SAVED_WORKFLOWS_DIR, f"{name}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

@app.delete("/workflows/{name}")
def delete_workflow(name: str):
    """Deletes a saved workflow."""
    file_path = os.path.join(SAVED_WORKFLOWS_DIR, f"{name}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "deleted", "name": name}
    raise HTTPException(status_code=404, detail="Workflow not found")


FEED_MANAGER = FeedManager()

# ... (run_inference_dag updates below) ...



# --- multiprocessing ---
from concurrent.futures import ProcessPoolExecutor

PROCESS_POOL = None # Initialized in main

def _run_feature_script_worker(script_path, input_df):
    """
    Worker function for ProcessPoolExecutor.
    Must be pure and picklable.
    """
    if not os.path.exists(script_path):
        return input_df
        
    try:
        spec = importlib.util.spec_from_file_location("feature_module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, "apply_features"):
            return module.apply_features(input_df.copy())
        else:
            return input_df
    except Exception as e:
        print(f"Error executing script {script_path}: {e}")
        return input_df

def _train_model_worker(model_name, input_df, models_dir):
    """
    Worker function for parallel model training.
    """
    try:
        # 1. Initialize Registry in this isolated process
        # We need to import classes fresh here
        from model_registry import ModelRegistry
        registry = ModelRegistry(models_dir)
        
        # 2. Train
        print(f"Worker: Starting training for {model_name}...")
        success = registry.train(model_name, input_df)
        
        return {
            "model": model_name, 
            "success": success, 
            "msg": "Training Complete" if success else "Training Failed"
        }
    except Exception as e:
        print(f"Worker Error training {model_name}: {e}")
        return {
            "model": model_name, 
            "success": False, 
            "msg": f"Error: {str(e)}"
        }

def execute_feature_script(script_name, input_df):
    """
    Legacy wrapper. In async parallel mode, we will call _run_feature_script_worker directly via executor.
    """
    script_path = os.path.join(FEATURE_SCRIPTS_DIR, script_name)
    return _run_feature_script_worker(script_path, input_df)

# Track next scheduled fetch time per context: {(pair, timeframe): next_fetch_timestamp}
NEXT_FETCH_SCHEDULE = {}

async def data_feed_loop():
    print("Starting Data Feed Loop...")
    while True:
        try:
             nodes = WORKFLOW_CONFIG.get("nodes", [])
             
             if not nodes:
                 await asyncio.sleep(2)
                 continue
                 
             # Simple Scan for Active Feeds
             active_contexts = set()
             feed_nodes = [n for n in nodes if n['type'] in ['liveDataFeed', 'liveDataFeedNode']]
             
             for n in feed_nodes:
                 pair = n['data'].get('pair', 'XBTUSD')
                 timeframe = n['data'].get('timeframe', '1m')
                 active_contexts.add((pair, timeframe))
            
             current_time = time.time()
             data_fetched_any = False
             has_instant_replay = False # [NEW] Track if we need high-speed looping
             
             # --- 3.0 Handle Data Replayers ---
             # If GLOBAL PAUSE is active, we just skip all replayer advancement steps.
             # We still need to populate stats (so UI shows "Paused"), but not advance.
             
             replayer_nodes = [n for n in nodes if n['type'] == 'dataReplayer']
             for n in replayer_nodes:

                 node_id = n['id']
                 pair = n['data'].get('pair', 'REPLAY')
                 timeframe = n['data'].get('timeframe', '1m')
                 dataset_name = n['data'].get('dataset')
                 frequency_str = n['data'].get('frequency', '5s')
                 
                 # 0. Parse Frequency
                 freq_map = {'instant': 0, '1s': 1, '5s': 5, '10s': 10, '30s': 30, '1m': 60}
                 freq_sec = freq_map.get(frequency_str, 5)
                 
                 if freq_sec == 0:
                     has_instant_replay = True # [NEW]
                 
                 # 1. Initialize Replayer if needed

                 if node_id not in REPLAY_INSTANCES:
                     if dataset_name:
                         csv_path = os.path.join(DATASETS_DIR, dataset_name)
                         replayer = DataReplayer(csv_path)
                         if replayer.load_dataset():
                            # [FIX] Start at index 719 to ensure we have a full 720-window immediately
                            if len(replayer.df) > 720:
                                replayer.current_index = 719
                            
                            REPLAY_INSTANCES[node_id] = {
                                "instance": replayer,
                                "next_run": current_time + 1, # Start soon
                                "freq": freq_sec,
                                "is_paused": False,
                                "freq": freq_sec,
                                "is_paused": False,
                                "start_time": current_time,
                                "end_time": None
                            }
                            # Clear existing buffer
                            FEED_MANAGER.raw_buffers.pop(f"{pair}_{timeframe}", None)
                         else:
                            print(f"Failed to load dataset {dataset_name} for node {node_id}")
                 
                 # 2. Check Schedule
                 if node_id in REPLAY_INSTANCES:
                     REPLAY_INSTANCES[node_id]["freq"] = freq_sec  # [FIX] Update frequency dynamically
                     state = REPLAY_INSTANCES[node_id]
                     
                     # Update Status
                     rep_inst = state["instance"]
                     total_rows = len(rep_inst.df) if rep_inst.df is not None else 0
                     curr_idx = rep_inst.current_index
                     prog = (curr_idx / total_rows * 100) if total_rows > 0 else 0
                     # Check if finished right now to freeze timer immediately
                     if not rep_inst.has_more() and state.get("end_time") is None:
                         state["end_time"] = time.time()
                     
                     start_ts = state.get("start_time", current_time) # [FIX] Restore missing definition
                     end_ts = state.get("end_time")
                     elapsed_val = (end_ts - start_ts) if end_ts else (current_time - start_ts)

                     REPLAY_STATS[node_id] = {
                         "current": curr_idx,
                         "total": total_rows,
                         "progress": prog,
                         "is_paused": IS_PAUSED, # [MOD] Use global pause
                         "is_finished": end_ts is not None,
                         "elapsed": elapsed_val
                     }

                     # [NEW] Add Current Timestamp to Stats
                     if curr_idx > 0 and rep_inst.df is not None:
                         try:
                             # Get timestamp of the LAST sent item (current_index - 1)
                             # Check bounds
                             read_idx = min(curr_idx - 1, len(rep_inst.df) - 1)
                             ts_val = rep_inst.df.index[read_idx]
                            
                             # [FIX] Handle NAType/None
                             if pd.isna(ts_val):
                                 REPLAY_STATS[node_id]["current_timestamp"] = None
                             else:
                                 # Convert to float/int if it's a Timestamp object
                                 if hasattr(ts_val, 'timestamp'):
                                     ts_val = ts_val.timestamp()
                                     
                                 REPLAY_STATS[node_id]["current_timestamp"] = ts_val
                         except Exception as e:
                             print(f"Error getting replay timestamp: {e}")
                     
                     if IS_PAUSED:
                         # If paused, update next_run to stay in the future so it doesn't backlog
                         state["next_run"] = current_time + 1
                         continue

                     if current_time >= state["next_run"]:
                         replayer = state["instance"]

                         if replayer.has_more():
                             # [MODIFIED] Windowed Replay (User Request 6)
                             # Pass a window of 720 rows, step of 1
                             
                             # 1. Advance Index
                             # replayer.next_row() returns single, but we want to simulate the whole window manually?
                             # Or use replayer to track index, and slice DF.
                             
                             # We'll use the replayer internal state
                             idx = replayer.current_index
                             
                             # Slice: [max(0, idx - 720 + 1) : idx + 1]
                             start_idx = max(0, idx - 719)
                             end_idx = idx + 1
                             
                             window_df = replayer.df.iloc[start_idx:end_idx].copy()
                             
                             # Advance Replayer State
                             replayer.current_index += 1
                             
                             # Inject into Feed Manager using SET (Replace)
                             FEED_MANAGER.set_buffer(pair, timeframe, window_df)
                                 
                             # NOTE: We add to active_contexts so process_node works
                             active_contexts.add((pair, timeframe))
                             data_fetched_any = True
                                 
                             # Schedule next
                             state["next_run"] = current_time + state["freq"]
                         else:
                             # Finished
                             if state.get("end_time") is None:
                                 state["end_time"] = time.time()
                             pass 

             # Identify contexts managed by Replayer to exclude from Live Fetch
             replay_contexts = set()
             for n in replayer_nodes:
                  p = n['data'].get('pair', 'REPLAY')
                  tf = n['data'].get('timeframe', '1m')
                  replay_contexts.add((p, tf)) 

             for pair, timeframe in active_contexts:
                 # Determine scheduling
                 should_fetch = False
                 sched_key = (pair, timeframe)
                 
                 # Get interval in seconds
                 minutes = KRAKEN_FEED.interval_map.get(timeframe, 1)
                 interval_seconds = minutes * 60
                 
                 if sched_key not in NEXT_FETCH_SCHEDULE:
                     should_fetch = True
                 elif current_time >= NEXT_FETCH_SCHEDULE[sched_key]:
                     should_fetch = True
                 
                 # SKIP if this context is driven by a Replayer
                 if (pair, timeframe) in replay_contexts:
                     should_fetch = False
                 
                 # [MODIFIED] Global Pause check for Live Feed (User Request 3)
                 if IS_PAUSED:
                     should_fetch = False

                 if should_fetch:
                     # Calculate next schedule (next candle boundary + 2 seconds)
                     next_boundary = ((int(current_time) // interval_seconds) + 1) * interval_seconds
                     NEXT_FETCH_SCHEDULE[sched_key] = next_boundary + 2
                     
                     new_data = KRAKEN_FEED.fetch_ohlcv(pair, timeframe)
                     if new_data is not None:
                         # [MODIFIED] Stateless Live Feed (User Request 1)
                         # Use set_buffer instead of update_buffer
                         FEED_MANAGER.set_buffer(pair, timeframe, new_data)
                         data_fetched_any = True
            
             if data_fetched_any:
                 # Rebuild UNIFIED DATASET (optional, but good for /dataset view)
                 FEED_MANAGER.rebuild_unified_dataset(list(active_contexts))
                 
                 # Update global snapshot for UI
                 LATEST_DATA["FEED_SNAPSHOT"] = convert_numpy(FEED_MANAGER.get_snapshot())
                 LATEST_DATA["ACTIVE_FEEDS"] = [f"{p}_{t}" for p, t in active_contexts]
                 
                 run_inference_dag_task = asyncio.create_task(run_inference_dag())
                 await run_inference_dag_task
             
        except Exception as e:
            print(f"Error in data_feed_loop: {e}")
        
        
        # Check more frequently (0.1s) to catch the scheduled times closely for fast replay
        # If instant replay is active, allow 0-delay yielding (bypasses Windows 15ms timer resolution)
        sleep_time = 0 if has_instant_replay else 0.1
        await asyncio.sleep(sleep_time)


def build_execution_order(nodes, edges):
    """
    Performs topological sort to determine execution order.
    Returns a list of nodes in execution order.
    """
    if not nodes:
        return []

    node_map = {n['id']: n for n in nodes}
    adj_list = {n['id']: [] for n in nodes}
    in_degree = {n['id']: 0 for n in nodes}
    
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        if src in adj_list and tgt in in_degree:
            adj_list[src].append(tgt)
            in_degree[tgt] += 1
            
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_nodes = []
    
    while queue:
        curr_id = queue.pop(0)
        sorted_nodes.append(node_map[curr_id])
        
        for neighbor in adj_list[curr_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    if len(sorted_nodes) != len(nodes):
        print("Warning: Cycle detected or disconnected graph issues. Execution might be partial.")
        
    return sorted_nodes

def execute_dynamic_condition(code_str, data_package):
    """
    Executes user defined python code for If/Else node.
    Wraps code in a function to allow 'return True/False'.
    """
    try:
        # Indent user code
        indented_code = "\n    ".join(code_str.splitlines())
        wrapper_code = f"def user_logic(data_package):\n    {indented_code}"
        
        local_scope = {}
        # Safe-ish exec: no globals access by default beyond builtins, but user has full local control
        exec(wrapper_code, {"print": print, "len": len, "float": float, "int": int, "str": str}, local_scope)
        
        if "user_logic" in local_scope:
            return local_scope["user_logic"](data_package), None
        return False, "Function user_logic definition failed"
    except Exception as e:
        print(f"Condition Execution Error: {e}")
        return False, str(e)

def process_node(node, inputs):
    """
    Generic node processor.
    inputs: List of data objects from upstream nodes.
    """
    node_type = node['type']
    node_id = node['id']
    
    global IS_PAUSED # [NEW] Access global pause state

    # --- 1. Live Data Feed ---
    if node_type in ['liveDataFeed', 'liveDataFeedNode']:
        # Doesn't use upstream inputs, fetches from Global Feed Manager
        pair = node['data'].get('pair', 'XBTUSD')
        timeframe = node['data'].get('timeframe', '1m')
        key = f"{pair}_{timeframe}"
        df = FEED_MANAGER.raw_buffers.get(key)
        # [STANDARD] Wrap in Data Package
        if df is not None:
             return {"data": df, "ts": time.time(), "pair": pair}
        return None
        
    # --- 1.5 Data Replayer ---
    elif node_type == 'dataReplayer':
        pair = node['data'].get('pair', 'REPLAY')
        timeframe = node['data'].get('timeframe', '1m')
        key = f"{pair}_{timeframe}"
        df = FEED_MANAGER.raw_buffers.get(key)
        
        if df is not None and not df.empty:
             # [FIX] Use the actual timestamp from the data (last row), not wall clock
             # This ensures downstream nodes (Broker) get the correct historical time.
             last_ts = df.index[-1]
             if isinstance(last_ts, pd.Timestamp):
                 last_ts = last_ts.timestamp()
             
             return {"data": df, "ts": last_ts, "pair": pair, "source": "replayer"}
        return None

    # --- 2. Feature Engineering ---
    elif node_type == 'featureEngineering':
        print(f"DEBUG: Processing FeatureEngineering {node_id}")
        input_pkg = None
        for i in inputs:
            print(f"DEBUG: FE input type {type(i)}")
            if isinstance(i, dict) and 'data' in i:
                if isinstance(i['data'], pd.DataFrame):
                    input_pkg = i
                    break
            elif isinstance(i, pd.DataFrame) and not i.empty:
                input_pkg = {"data": i}
                break
        
        if input_pkg is None:
            return None
            
        input_df = input_pkg['data']
        script_name = node['data'].get('scriptName')
        
        result_df = input_df
        error_msg = None
        
        if script_name:
            print(f"DEBUG: Executing script {script_name}...")
            try:
                result_df = execute_feature_script(script_name, input_df)
                if result_df.empty:
                    error_msg = "Not enough data (Empty Result)"
                elif len(input_df) < 5:
                     pass 
            except Exception as e:
                error_msg = f"Script Error: {str(e)}"
            
            print(f"DEBUG: Script complete. Result rows: {len(result_df)}")
            
        if error_msg:
             return {"error": error_msg}

        output_pkg = input_pkg.copy()
        output_pkg['data'] = result_df
        return output_pkg

    # --- 2.1 If/Else Node ---
    elif node_type == 'ifElseNode':
        input_pkg = None
        for i in inputs:
            if isinstance(i, dict) and 'data' in i:
                input_pkg = i
                break
        
        if input_pkg is None:
            return None
            
        code = node['data'].get('code', 'return True')
        condition_met, error_msg = execute_dynamic_condition(code, input_pkg)
        
        output_pkg = input_pkg.copy()
        
        if error_msg:
             with open("debug_nodes.txt", "a") as f:
                 f.write(f"If/Else Node {node_id} Error: {error_msg}. Pausing: {True}\n")
             
             output_pkg['condition_met'] = None
             output_pkg['error'] = error_msg
             IS_PAUSED = True
             return output_pkg
        
        with open("debug_nodes.txt", "a") as f:
             f.write(f"If/Else Node {node_id} Success. Result: {condition_met}\n")

        try:
            if isinstance(condition_met, (pd.DataFrame, pd.Series, np.ndarray)):
                if hasattr(condition_met, 'empty'):
                     safe_bool = not condition_met.empty
                elif hasattr(condition_met, 'size'):
                     safe_bool = condition_met.size > 0
                else:
                     safe_bool = bool(condition_met)
            else:
                safe_bool = bool(condition_met)
        except Exception as e:
            print(f"If/Else Node {node_id} boolean conversion error: {e}")
            safe_bool = False
            output_pkg['error'] = f"Type Conversion Error: {str(e)}"
            
        output_pkg['condition_met'] = safe_bool
        return output_pkg

    # --- 2.2 Random Action ---
    elif node_type == 'randomAction':
        # Generate random trading signal 
        # Actions: buy, sell, pass
        # Volume: 0.1 - 1.0
        
        actions = ['buy', 'sell', 'pass'] # [FIX] Reduced bias to ensure activity
        chosen_action = random.choice(actions)
        
        # Determine volume if not pass
        vol = 0.0
        if chosen_action != 'pass':
            vol = round(random.uniform(0.1, 1.0), 2)
            
        # [FIX] Propagate Timestamp info from Input
        current_ts = time.time()
        input_ts_found = False
        
        # Try to find timestamp in inputs
        input_ts_found = False
        
        # [FIX] Prioritize explicit 'ts' key from inputs (e.g. from Replayer)
        for i in inputs:
             # Check top level dict for 'ts'
             if isinstance(i, dict) and 'ts' in i:
                 current_ts = i['ts']
                 input_ts_found = True
                 break

             # Check inside 'data' if it's a dict with 'ts' (unlikely but possible)
             if isinstance(i, dict) and 'data' in i:
                 inner_data = i['data']
                 if isinstance(inner_data, dict) and 'ts' in inner_data:
                     current_ts = inner_data['ts']
                     input_ts_found = True
                     break
                 
                 # Case 1: Inner data is DataFrame -> Use Index
                 if isinstance(inner_data, pd.DataFrame) and not inner_data.empty:
                     current_ts = inner_data.index[-1]
                     input_ts_found = True
                     break
                     
             # Direct DataFrame
             if isinstance(i, pd.DataFrame) and not i.empty:
                 current_ts = i.index[-1]
                 input_ts_found = True
                 break

        # [FIX] Propagate Price/Close from Inputs for Broker Fallback
        input_price = None
        for i in inputs:
             val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
             if isinstance(val, pd.DataFrame) and not val.empty:
                 if 'close' in val.columns:
                     input_price = val.iloc[-1]['close']
                 elif 'price' in val.columns:
                     input_price = val.iloc[-1]['price']
                 if input_price is not None:
                     break
            
        return {
            "data": { # Wrap Action result in 'data'
                "action": chosen_action,
                "volume": vol,
                "price": input_price # [FIX] Pass price
            },
            "ts": current_ts,
            "action": chosen_action # Convenience top-level
        }

    # --- 2.5 Trading Brain ---
    elif node_type == 'tradingBrain':
        # Input: Expects a DataFrame (usually with prediction col) OR a dict from previous node
        # We also need the current price (close).
        # We will package the LAST row of the input DF into a dictionary.
        
        input_data = {}
        timestamp_source = None
        
        # 1. Try to find dataframe inputs
        dfs = []
        for i in inputs:
            # [STANDARD] Unwrap
            val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
            if isinstance(val, pd.DataFrame) and not val.empty:
                dfs.append(val)
        
        if dfs:
            # Take the latest row of the first DF
            latest = dfs[0].iloc[-1].to_dict()
            input_data.update(latest)
            timestamp_source = dfs[0].index[-1]
            
        # Also check dict inputs for 'ts'
        if timestamp_source is None:
             for i in inputs:
                 if isinstance(i, dict) and 'ts' in i:
                     timestamp_source = i['ts']
                     break
        
        if timestamp_source is None:
            timestamp_source = time.time()

        script_name = node['data'].get('scriptName')
        if not script_name:
            return {"error": "No script selected"}
            
        script_path = os.path.join(TRADING_BRAINS_DIR, script_name)
        if not os.path.exists(script_path):
            return {"error": f"Script {script_name} not found"}
            
        try:
            spec = importlib.util.spec_from_file_location("brain_module", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, "decide_action"):
                # Execute Logic
                action_result = module.decide_action(input_data)
                
                # [FIX] Enforce Timestamp Propagation
                if isinstance(action_result, dict):
                    if 'ts' not in action_result:
                        action_result['ts'] = timestamp_source
                
                # [STANDARD] Return package
                return {
                    "data": action_result,
                    "ts": timestamp_source,
                    **action_result # Merge top level for compat
                }
            else:
                return {"error": "Script must implement decide_action(inputs)"}
        except Exception as e:
            return {"error": f"Brain execution error: {str(e)}"}

    # --- 2.6 Broker Node (Simulated Kraken) ---
    elif node_type == 'brokerNode':
        # Input: Expects "action" output from TradingBrain/RandomAction
        mode = node['data'].get("mode", "live")
        
        trade_signal = None
        market_price = None
        current_ts = None
        input_df_price = None # [NEW] Capture input price
        
        # Parse inputs
        for i in inputs:
            # [STANDARD] Unwrap
            val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
            
            # Check for action dict
            if isinstance(val, dict):
                if "action" in val:
                    trade_signal = val
            
            # [FIX] Check for DataFrame with action
            if isinstance(val, pd.DataFrame) and not val.empty:
                # We take the LAST row
                last_row = val.iloc[-1]
                if 'action' in val.columns:
                     # Construct trade signal from DF row
                     signal = {'action': last_row['action']}
                     if 'volume' in val.columns:
                         signal['volume'] = last_row['volume']
                     if 'price' in val.columns:
                         signal['price'] = last_row['price']
                     elif 'close' in val.columns:
                         signal['price'] = last_row['close']
                         
                     trade_signal = signal

            # Check for TS
            if isinstance(val, dict) and "ts" in val:
                 current_ts = val["ts"]
            elif isinstance(i, dict) and "ts" in i:
                current_ts = i["ts"]

            # [FIX] Check for Price in Dict inputs (from RandomAction)
            if isinstance(val, dict):
                if 'price' in val:
                    input_df_price = val['price']
                elif 'close' in val:
                    input_df_price = val['close']

            # [NEW] Check for DataFrame with price
            if isinstance(val, pd.DataFrame) and not val.empty:
                if 'close' in val.columns:
                    input_df_price = val.iloc[-1]['close']
                elif 'price' in val.columns:
                    input_df_price = val.iloc[-1]['price']
                    
        # Execution Logic (Stateless)
        # 1. Price Discovery
        dataset_name = node['data'].get("dataset")
        market_price = None
        
        if dataset_name and current_ts is not None:
            # Debug Input Structure
            try:
                with open("debug_nodes.txt", "a") as f:
                    # Convert inputs to string safely
                    debug_inputs = []
                    for inp in inputs:
                         if isinstance(inp, dict):
                              debug_inputs.append(str(inp)[:200]) # Cap length
                         elif isinstance(inp, pd.DataFrame):
                              debug_inputs.append(f"DF shape={inp.shape} cols={list(inp.columns)}")
                         else:
                              debug_inputs.append(str(inp))
                    f.write(f"BROKER_INPUT_DEBUG: Inputs={debug_inputs}\\n")
            except Exception as e:
                 print(f"Debug log error: {e}")

            # Simple Caching
            if 'DATASET_CACHE' not in globals():
                globals()['DATASET_CACHE'] = {}
            
            if dataset_name not in globals()['DATASET_CACHE']:
                 fpath = os.path.join(DATASETS_DIR, dataset_name)
                 if os.path.exists(fpath):
                     globals()['DATASET_CACHE'][dataset_name] = pd.read_csv(fpath)
                     # Pre-process index
                     if 'timestamp' in globals()['DATASET_CACHE'][dataset_name].columns:
                          # Try parsing
                          try:
                               # Check if numeric (Unix Timestamp)
                               first_val = globals()['DATASET_CACHE'][dataset_name]['timestamp'].iloc[0]
                               if isinstance(first_val, (int, float, np.integer, np.floating)):
                                   # Assume seconds if < 3e10 (year 2920), else ms/ns
                                   # Simple heuristic for our data
                                   if first_val < 3e10: 
                                        globals()['DATASET_CACHE'][dataset_name]['timestamp'] = pd.to_datetime(globals()['DATASET_CACHE'][dataset_name]['timestamp'], unit='s')
                                   else:
                                        globals()['DATASET_CACHE'][dataset_name]['timestamp'] = pd.to_datetime(globals()['DATASET_CACHE'][dataset_name]['timestamp'], unit='ms')
                               else:
                                   # String or other
                                   globals()['DATASET_CACHE'][dataset_name]['timestamp'] = pd.to_datetime(globals()['DATASET_CACHE'][dataset_name]['timestamp'])
                          except Exception as e:
                               print(f"Dataset Timestamp Parse Error: {e}")
                          
                          globals()['DATASET_CACHE'][dataset_name].set_index('timestamp', inplace=True)
                          globals()['DATASET_CACHE'][dataset_name].sort_index(inplace=True)
            
            df_cache = globals()['DATASET_CACHE'].get(dataset_name)
            if df_cache is not None and not df_cache.empty:
                try:
                    # Normalize current_ts
                    ts_obj = None
                    if isinstance(current_ts, (int, float)):
                        ts_obj = pd.to_datetime(current_ts, unit='s')
                    else:
                        ts_obj = pd.to_datetime(current_ts)
                    
                    # DEBUG LOGGING
                    with open("debug_nodes.txt", "a") as f:
                         f.write(f"Broker Debug: Input TS={current_ts}, Converted={ts_obj}, IndexType={df_cache.index.dtype}, SampleIndex={df_cache.index[0]}\\n")

                    # Find exact match or nearest
                    # We need the INTEGER location to shift forward
                    if ts_obj in df_cache.index:
                         # Get integer location
                         idx_loc = df_cache.index.get_loc(ts_obj)
                         
                         # Check for duplicates or slice
                         if isinstance(idx_loc, slice):
                             idx_loc = idx_loc.start
                         elif isinstance(idx_loc, (np.ndarray, list)):
                             idx_loc = idx_loc[0]
                         
                         with open("debug_nodes.txt", "a") as f:
                             f.write(f"BROKER_DEBUG: InputTS={ts_obj} FoundAt={idx_loc} Next={idx_loc+1}\n")

                         # Check if next candle exists
                         if idx_loc + 1 < len(df_cache):
                              # Execute at t+1 OPEN
                              next_candle = df_cache.iloc[idx_loc + 1]
                              market_price = next_candle['open']
                              exec_ts = df_cache.index[idx_loc + 1] # Use t+1 timestamp for record
                              
                              with open("debug_nodes.txt", "a") as f:
                                  f.write(f"BROKER_DEBUG: Executing at TS={exec_ts} Price={market_price}\n")
                         else:
                              # End of Data
                              return {"status": f"Wait: End of Data at {current_ts}", "action": "wait"}
                    else:
                         # Nearest match fallback (rare)
                         idx_loc = df_cache.index.searchsorted(ts_obj)
                         with open("debug_nodes.txt", "a") as f:
                             f.write(f"BROKER_DEBUG: InputTS={ts_obj} Not Found. SearchSorted={idx_loc}\n")
                             
                         if idx_loc + 1 < len(df_cache):
                              next_candle = df_cache.iloc[idx_loc + 1]
                              market_price = next_candle['open']
                              exec_ts = df_cache.index[idx_loc + 1]
                         else:
                              # [FIX] Try fallback to input price if available
                              if input_df_price is not None:
                                  market_price = input_df_price
                                  exec_ts = ts_obj
                              else:
                                  return {"status": f"Wait: Timestamp {current_ts} beyond dataset end.", "action": "wait"}
                    
                    if isinstance(market_price, pd.Series):
                        market_price = market_price.iloc[-1]
                except Exception as e:
                    print(f"Broker Price Lookup Error: {e}")
                    with open("debug_nodes.txt", "a") as f:
                        f.write(f"BROKER_DEBUG: Error {e}\n")

        if market_price is None:
             # Can't trade without price
             return {"status": f"No Price for {current_ts}", "action": "wait"}

        if not trade_signal:
             return {"status": "Waiting for orders", "action": "wait"}

        # 2. Execute Order (Simulated)

        action = trade_signal.get('action', 'pass').lower()
        volume = float(trade_signal.get('volume', 0.0))
        fee_rate = 0.0026 # 0.26% Taker
        
        # Ensure timestamp is float (Unix Seconds) for Frontend
        exec_ts = current_ts
        if hasattr(exec_ts, 'timestamp'):
             exec_ts = exec_ts.timestamp()
        
        execution_result = {
            "timestamp": exec_ts,
            "action": action,
            "price": market_price,
            "volume": volume,
            "dataset": dataset_name
        }
        
        if action == 'buy':
            cost_raw = volume * market_price
            fee = cost_raw * fee_rate
            total_cost = cost_raw + fee
            
            execution_result.update({
                "fee": fee,
                "cost": total_cost,
                "value": cost_raw
            })
                 
        elif action == 'sell':
            revenue_raw = volume * market_price
            fee = revenue_raw * fee_rate
            total_revenue = revenue_raw - fee
            
            execution_result.update({
                "fee": fee,
                "revenue": total_revenue,
                "value": revenue_raw
            })
        # Return as DataFrame for Data Store compatibility
        # 3. Format Output
        # Create single-row DataFrame
        df_out = pd.DataFrame([execution_result])
        
        # [FIX] STRICTLY Enforce DatetimeIndex
        try:
            if 'timestamp' in df_out.columns:
                # distinct conversion for robustness
                ts_col = df_out['timestamp']
                
                # Check if it needs conversion
                if pd.api.types.is_numeric_dtype(ts_col):
                     df_out['timestamp'] = pd.to_datetime(ts_col, unit='s')
                else:
                     df_out['timestamp'] = pd.to_datetime(ts_col)
                
                # Set Index explicitly
                df_out.set_index('timestamp', inplace=True, drop=False)
                
                # Verify Index Type
                if not isinstance(df_out.index, pd.DatetimeIndex):
                     with open("debug_nodes.txt", "a") as f:
                         f.write(f"BROKER_ERROR: Failed to set DatetimeIndex. Got {type(df_out.index)}\n")
            else:
                 # Fallback if no timestamp provided (shouldn't happen with logic above)
                 df_out.index = pd.to_datetime([time.time()], unit='s')
                 df_out['timestamp'] = df_out.index
                 
            # Debug Log for Verification
            # with open("debug_nodes.txt", "a") as f:
            #     f.write(f"BROKER_DEBUG: Output Index={df_out.index} Cols={df_out.columns}\n")
                 
        except Exception as e:
            print(f"Broker Index formatting error: {e}")
            with open("debug_nodes.txt", "a") as f:
                 f.write(f"BROKER_EXCEPTION: {e}\n")

        return {"data": df_out}

    # --- 2.7 Eval Node ---
    elif node_type == 'evalNode':
        # Input: DataFrame with filled orders
        input_df = None
        for i in inputs:
             val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
             if isinstance(val, pd.DataFrame):
                 input_df = val
                 break
        
        if input_df is None or input_df.empty:
            return {"status": "Waiting for Trade Data"}
            
        df = input_df.copy()
        required_cols = ['action', 'price', 'volume']
        if not all(c in df.columns for c in required_cols):
             return {"error": "Invalid Data Format. Missing action/price/volume."}
             
        # Trade Logic (FIFO)
        trades = []
        inventory = [] 
        
        for idx, row in df.iterrows():
            action = row['action']
            price = float(row['price'])
            vol = float(row['volume'])
            
            if action == 'buy':
                inventory.append({'price': price, 'vol': vol})
            elif action == 'sell':
                qty_to_sell = vol
                cost_basis = 0
                qty_filled = 0
                
                while qty_to_sell > 0 and inventory:
                    batch = inventory[0]
                    match_qty = min(batch['vol'], qty_to_sell)
                    
                    cost_basis += match_qty * batch['price']
                    qty_filled += match_qty
                    
                    batch['vol'] -= match_qty
                    qty_to_sell -= match_qty
                    
                    if batch['vol'] <= 0.00000001:
                        inventory.pop(0)
                        
                if qty_filled > 0:
                    avg_entry = cost_basis / qty_filled
                    ret = (price - avg_entry) / avg_entry
                    trades.append(ret)
                    
        # Binning
        if not trades:
            return {"distribution": [], "metrics": {}}

        trades_arr = np.array(trades)
        hist, bin_edges = np.histogram(trades_arr, bins=10)
        
        dist_data = []
        for i in range(len(hist)):
             dist_data.append({
                 "range_start": float(bin_edges[i]),
                 "range_end": float(bin_edges[i+1]),
                 "count": int(hist[i])
             })
             
        metrics = {
             "total_trades": len(trades),
             "win_rate": float(np.mean(trades_arr > 0)),
             "avg_return": float(np.mean(trades_arr)),
        }
        
        return {
            "distribution": dist_data,
            "metrics": metrics
        }

        


    # --- 3. Merge Node ---
    elif node_type == 'mergeNode':
        # Strictly expect DataFrames
        dfs = [i for i in inputs if isinstance(i, pd.DataFrame) and not i.empty]
        
        if not dfs:
            return {"status": "Waiting for valid DataFrame inputs..."}

        merge_type = node['data'].get('mergeType', 'concat')
        
        try:
            if merge_type == 'concat':
                # Axis 1 concat (side by side), aligning on index
                result = pd.concat(dfs, axis=1)
                # Deduplicate columns if any (though for meta-model we want distinct columns)
                # conversion to dict later handles it, but let's keep it clean
                return result
            elif merge_type == 'inner_join':
                result = dfs[0]
                for other in dfs[1:]:
                    result = result.join(other, how='inner', lsuffix='_L', rsuffix='_R')
                return result
            elif merge_type == 'outer_join':
                result = dfs[0]
                for other in dfs[1:]:
                    result = result.join(other, how='outer', lsuffix='_L', rsuffix='_R')
                return result
            else:
                return pd.concat(dfs, axis=1) # Default
        except Exception as e:
            return f"Merge Failed: {str(e)}"

    # --- 4. ML Model ---
    elif node_type in ['model', 'modelNode']:
        model_name = node['data'].get('modelName')
        if not model_name:
            return "No Model Selected"
            
        # Expects DataFrame (to take last row)
        input_df = None
        for i in inputs:
             # [STANDARD] Unwrap
             val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
             
             if isinstance(val, pd.DataFrame) and not val.empty:
                input_df = val
                break
                
        if input_df is None:
            return "Waiting for Data..."
            
        # Prediction Logic
        # We pass the FULL DATAFRAME to the model so it can compute features (lags, etc.)
        # checking the last row for validity first
        last_row_df = input_df.iloc[[-1]]
        
        try:
            # Check for NaNs in the input row (raw data)
            # [MODIFIED] Removed strict NaN check. 
            # DataStore inputs are sparse (e.g. action=NaN when just price update).
            # We let the model wrapper extract specific features and handle validity.
            # if last_row_df.isnull().values.any():
            #      with open("debug_nodes.txt", "a") as f:
            #          f.write(f"Model {model_name}: Input contains NaN\\n")
            #      return "Error: Input contains NaN"
            
            # Pass FULL DataFrame to REGISTRY.predict
            val = REGISTRY.predict(model_name, input_df)
            
            if val is None:
                 with open("debug_nodes.txt", "a") as f:
                     f.write(f"Model {model_name}: Registry returned None (Model not found or failed)\\n")
                 return "Error: Model not found in registry"

            # Val is likely an array of predictions matching the input df length.
            # We only care about the last one.
            if hasattr(val, 'tolist') or isinstance(val, (list, np.ndarray)):
                 val_array = np.array(val)
                 if val_array.size > 0:
                     val = val_array[-1]
                 else:
                     return "Error: Empty prediction output"
                     
            if hasattr(val, 'item'):
                 val = val.item()

            # Simple check for scalar errors (NaN/Inf)
            if isinstance(val, (int, float)) and (pd.isna(val) or np.isinf(val)):
                 with open("debug_nodes.txt", "a") as f:
                     f.write(f"Model {model_name}: Returned NaN\\n")
                 return "Error: Model returned NaN"
            
            # For UI: Update LATEST_INFERENCE_RESULTS with the scalar
            LATEST_INFERENCE_RESULTS[node_id] = convert_numpy(val)
            
            # For Data Flow: Return a DataFrame with 1 row (the last prediction)
            col_name = model_name if model_name else "prediction"
            if col_name.endswith(".pkl"):
                col_name = col_name[:-4]
            output_df = pd.DataFrame({col_name: [val]}, index=last_row_df.index)
            
            return output_df
            
        except Exception as e:
            with open("debug_nodes.txt", "a") as f:
                f.write(f"Model {model_name} Error: {str(e)}\\n")
            return f"Error: {str(e)}"

    # --- 5. Data Store Node (Write & Read) ---
    elif node_type == 'dataStore':
        # [PERSISTENCE MODE] Upsert inputs to Disk via Service
        input_data = None
        for i in inputs:
            if isinstance(i, dict) and 'data' in i:
                input_data = i['data']
                break
            if isinstance(i, pd.DataFrame):
                input_data = i
                break
        
        # Get configured filename (default to shared)
        filename = node['data'].get('filename', 'data_storage.parquet')
        
        # 1. WRITE (Upsert)
        if input_data is not None and isinstance(input_data, pd.DataFrame) and not input_data.empty:
             updated_df = DATA_STORE_SERVICE.update(input_data, filename=filename)
             return {"data": updated_df}
        
        # 2. READ (No new input, just return current state)
        current_data = DATA_STORE_SERVICE.load(filename=filename)
        if not current_data.empty:
             return {"data": current_data}
        
        # 3. Empty State
        return None

    # --- 6. Train Node ---
    elif node_type == 'trainNode':
        model_name = node['data'].get('modelName')
        if not model_name:
            return {"status": "No Model Selected"}
            
        # Expects Data Package with "data" key containing DataFrame
        input_data = None
        for i in inputs:
             # [STANDARD] Unwrap
             val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
             if isinstance(val, pd.DataFrame) and not val.empty:
                 input_data = val
                 break
        
        if input_data is None:
            return {"status": "Waiting for Data..."}
            
        # Trigger Training
        # We only want to train if we haven't trained recently? 
        # Or should we train on every pulse? 
        # USER REQUEST: "passed to this node... should be used to train"
        # Since this is a detailed operation, maybe we should check if data changed significantly?
        # For now, we will attempt training every time we get data, relying on Model Registry to handle logic?
        # WARNING: Training is expensive. If this flow runs every 3s, we will re-train every 3s.
        # Ideally, we should add a "Trigger" or only train if manually requested?
        # The user request implies: "data passed... used to train".
        # Let's add a simple check: Only train if we receive a *new* large chunk of data?
        # Or actually, the user likely will connect a Data Replayer (Full Dataset) -> Train Node.
        # This will happen one-shot or stream.
        
        try:
            success = REGISTRY.train(model_name, input_data)
            if success:
                return {"status": "Training Success", "timestamp": time.time()}
            else:
                return {"status": "Training Failed"}
        except Exception as e:
            return {"status": f"Error: {str(e)}"}
            
    return None

# --- Global Execution Stats ---
EXECUTION_STATS = {
    "workflow_total_time_ms": 0,
    "nodes": {} # node_id -> {last_execution_time_ms: float, status: 'idle' | 'executing'}
}

LAST_WORKFLOW_START_TIME = 0

def get_topological_generations(nodes, edges):
    """
    Groups nodes into generations where each generation can be executed in parallel.
    Generation 0: Nodes with no dependencies.
    Generation N: Nodes whose dependencies are all in Generations < N.
    """
    if not nodes:
        return []

    node_map = {n['id']: n for n in nodes}
    adj_list = {n['id']: [] for n in nodes}
    in_degree = {n['id']: 0 for n in nodes}
    
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        if src in adj_list and tgt in in_degree:
            adj_list[src].append(tgt)
            in_degree[tgt] += 1
            
    # Initial Generation: Nodes with in-degree 0
    current_gen = [nid for nid, deg in in_degree.items() if deg == 0]
    generations = []
    
    while current_gen:
        generations.append([node_map[nid] for nid in current_gen])
        next_gen = []
        
        # Simulate execution to find next available nodes
        # (Subtract in-degree for neighbors)
        # Note: We can't just modify 'in_degree' blindly if we want strict generations.
        # But for topological generations (Kahn's algo variant):
        # We process a snapshot of the current generation.
        
        # Actually, standard Kahn's algo processes nodes one by one. 
        # To get "Parallel Layers":
        # 1. Identify all nodes with in-degree 0 -> Gen 0
        # 2. Remove Gen 0 nodes and their edges.
        # 3. Identify new nodes with in-degree 0 -> Gen 1
        # ...
        
        temp_next_candidates = []
        for nid in current_gen:
            for neighbor in adj_list[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    temp_next_candidates.append(neighbor)
        
        current_gen = temp_next_candidates
        
    return generations

async def process_node_async(node, inputs):
    """
    Async wrapper.
    - Feature Engineering -> Process Pool (True Parallelism)
    - Others -> Thread Pool (Concurrency)
    """
    node_type = node['type']
    
    if node_type == 'featureEngineering':
        # 1. Extract Inputs (Similar to process_node logic)
        input_df = None
        for i in inputs:
            # [STANDARD] Unwrap
            val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
            
            if isinstance(val, pd.DataFrame) and not val.empty:
                input_df = val
                break
        
        if input_df is None:
            return None
            
        script_name = node['data'].get('scriptName')
        if script_name and PROCESS_POOL:
            # Offload to Process
            script_path = os.path.join(FEATURE_SCRIPTS_DIR, script_name)
            loop = asyncio.get_running_loop()
            try:
                # We must ensure arguments are simple (DF is picklable)
                return await loop.run_in_executor(PROCESS_POOL, _run_feature_script_worker, script_path, input_df)
            except Exception as e:
                print(f"Process Pool Error on {script_name}: {e}")
                return input_df
        else:
            # Fallback (No script or No Pool)
            return await asyncio.to_thread(process_node, node, inputs)

    elif node_type == 'trainNode':
        # SCRIPT-BASED TRAINING
        
        # 1. Get Scripts
        # Supports multiple scripts if needed, or just one
        script_names = node['data'].get('scriptNames', [])
        # Fallback for UI transition
        if not script_names and node['data'].get('scriptName'):
            script_names = [node['data']['scriptName']]
        
        # Legacy support: if only modelNames are present, map them?
        # For now, if no scriptNames, we should probably return a hint or error
        # unless we auto-map 'xgb_next_close' -> 'train_xgb_next_close.py'
        if not script_names:
             # Just for partial compatibility, check modelNames
             m_names = node['data'].get('modelNames', [])
             if m_names:
                 return {"status": "Legacy Mode Not Supported. Select a Script."}
             return {"status": "No Scripts Selected"}
             
        # 2. Get Data
        input_df = None
        for i in inputs:
            val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
            if isinstance(val, pd.DataFrame) and not val.empty:
                input_df = val
                break
        
        if input_df is None:
            return {"status": "Waiting for Data..."}
            
        # 3. Save Data to Temp Parquet
        # We need a temp file path that the subprocess can read
        TEMP_DATA_DIR = os.path.join(BASE_DIR, "temp_data")
        os.makedirs(TEMP_DATA_DIR, exist_ok=True)
        # Using timestamp to avoid collision (or uuid)
        temp_filename = f"train_input_{int(time.time() * 1000)}.parquet"
        temp_path = os.path.join(TEMP_DATA_DIR, temp_filename)
        
        try:
            # Drop index if it's default range index? No, preserve it.
            input_df.to_parquet(temp_path)
        except Exception as e:
            return {"status": f"Error saving temp data: {e}"}

        # 4. Run Scripts
        # We just run them sequentially for now in the thread pool, or use ProcessPool?
        # Subprocess is blocking, so we should run it in a thread/executor.
        
        import subprocess
        
        def _run_script(s_name, d_path):
            script_full_path = os.path.join(BASE_DIR, "training", s_name)
            if not os.path.exists(script_full_path):
                return {"script": s_name, "success": False, "msg": "Script not found"}
                
            cmd = [sys.executable, script_full_path, d_path]
            try:
                # Capture output
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600) # 10 min timeout
                if result.returncode == 0:
                    return {"script": s_name, "success": True, "msg": "Success", "output": result.stdout}
                else:
                    return {"script": s_name, "success": False, "msg": f"Failed (Exit {result.returncode})", "output": result.stderr}
            except Exception as e:
                return {"script": s_name, "success": False, "msg": str(e), "output": ""}

        loop = asyncio.get_running_loop()
        results = []
        
        print(f"Scheduling training scripts: {script_names}")
        
        for s_name in script_names:
             # Run in thread executor (subprocess is IO bound-ish but we wait)
             res = await loop.run_in_executor(None, _run_script, s_name, temp_path)
             results.append(res)
             
        # Cleanup
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass
            
        # 5. Process Results
        success_count = 0
        failed = []
        
        for res in results:
            if res['success']:
                success_count += 1
                # If using Registry, we might want to reload it?
                # The script saves to disk. Registry.reload_models() might be needed if Registry is long-lived.
                # But main.py re-initializes registry logic or we have a global one?
                # GLOBAL REGISTRY exists.
                if REGISTRY:
                    # We should probably force reload or specific reload?
                    # Registry.reload_models() re-scans.
                    REGISTRY.reload_models() 
            else:
                failed.append(res['script'])
                print(f"Training Script Error ({res['script']}):\n{res['output']}")
        
        status_msg = f"Ran {success_count}/{len(script_names)}"
        if failed:
            status_msg += f" (Failed: {', '.join(failed)})"
            return {"status": status_msg, "timestamp": time.time(), "details": results}
            
        return {"status": status_msg, "timestamp": time.time()}

    elif node_type in ['model', 'modelNode']:
        # PARALLEL INFERENCE
        model_names = node['data'].get('modelNames', [])
        # Legacy fallback
        if not model_names and node['data'].get('modelName'):
             model_names = [node['data']['modelName']]
        
        if len(model_names) > 1:
            # 1. Get Data
            input_df = None
            for i in inputs:
                val = i.get('data') if (isinstance(i, dict) and 'data' in i) else i
                if isinstance(val, pd.DataFrame) and not val.empty:
                    input_df = val
                    break
            
            if input_df is None:
                return "Waiting for Data..."

            # 2. Run Parallel Predictions (Threads are fine for inference usually, or use Process Pool?)
            # ProcessPool pickling overhead might be high for high-freq inference.
            # Let's try ThreadPool first (asyncio.to_thread).
            
            async def predict_single(m_name, df):
                # Wrapper to call sync registry
                # We need to handle exceptions here to not crash gather
                try:
                    # We can reuse process_node logic by creating a fake node dict?
                    # Or just call registry directly.
                    # Registry is safe?
                    return await asyncio.to_thread(REGISTRY.predict, m_name, df)
                except Exception as e:
                    return None

            tasks = [predict_single(name, input_df) for name in model_names]
            results = await asyncio.gather(*tasks)
            
            # 3. Concatenate
            last_row_df = input_df.iloc[[-1]]
            output_cols = {}
            
            inference_results_map = {} # For UI display
            
            for m_name, res in zip(model_names, results):
                val = res
                
                # Validation Logic from process_node
                if val is not None:
                    if hasattr(val, 'tolist') or isinstance(val, (list, np.ndarray)):
                         val_array = np.array(val)
                         if val_array.size > 0:
                             val = val_array[-1]
                         else:
                             val = None
                             
                    if hasattr(val, 'item'):
                         val = val.item()
                         
                    # Check NaN
                    if isinstance(val, (int, float)) and (pd.isna(val) or np.isinf(val)):
                         val = None
                
                # Store
                safe_name = m_name
                if safe_name.endswith(".pkl"):
                    safe_name = safe_name[:-4]
                    
                output_cols[safe_name] = val
                inference_results_map[m_name] = convert_numpy(val) if val is not None else "Error"

            # Create Result DataFrame
            # We want one row with multiple columns
            # If all are None?
            
            # Construct DataFrame from dict
            # { "ModelA": [100], "ModelB": [200] } relative to index
            out_data = {k: [v] for k, v in output_cols.items()}
            output_df = pd.DataFrame(out_data, index=last_row_df.index)
            
            # Update Global UI State
            # We can store a dict instead of scalar
            LATEST_INFERENCE_RESULTS[node['id']] = inference_results_map
            
            return output_df

        else:
            # Single model -> Fallback to synchronous process_node which handles it fine
            return await asyncio.to_thread(process_node, node, inputs)

    else:
        # Others -> Thread Pool
        return await asyncio.to_thread(process_node, node, inputs)


async def run_inference_dag():
    """Generic DAG Execution Engine (Parallel & Async)."""
    global LATEST_INFERENCE_RESULTS, WORKFLOW_CONFIG, LATEST_DATA, EXECUTION_STATS, LAST_WORKFLOW_START_TIME, FULL_NODE_OUTPUTS
    
    nodes = WORKFLOW_CONFIG.get("nodes", [])
    edges = WORKFLOW_CONFIG.get("edges", [])
    
    if not nodes:
        return

    # 0. Calculate Frequency
    current_start_time = time.time()
    if LAST_WORKFLOW_START_TIME > 0:
        interval_ms = (current_start_time - LAST_WORKFLOW_START_TIME) * 1000
        EXECUTION_STATS["workflow_interval_ms"] = round(interval_ms, 2)
    LAST_WORKFLOW_START_TIME = current_start_time

    # 1. Build Generations for Parallel Execution
    generations = get_topological_generations(nodes, edges)
    
    # 2. Execution Loop
    node_outputs = {} # node_id -> result
    
    # Map incoming edges for quick lookup
    node_map = {n['id']: n for n in nodes}
    incoming_map = {n['id']: [] for n in nodes}
    for edge in edges:
        if edge['target'] in incoming_map:
            incoming_map[edge['target']].append(edge)
            
    workflow_start_time = current_start_time
    
    for generation in generations:
        if not generation:
            continue

        # Execute Generation in Parallel
        
        async def execute_and_time(node_):
            nid_ = node_['id']
            
            # Update Status to Executing (Atomic-ish for stats)
            if nid_ not in EXECUTION_STATS["nodes"]:
                EXECUTION_STATS["nodes"][nid_] = {}
            EXECUTION_STATS["nodes"][nid_]["status"] = "executing"
            
            # Gather inputs (inside the task to be safe)
            incoming_edges_ = incoming_map.get(nid_, [])
            inputs_ = []
            for edge in incoming_edges_:
                sid = edge.get('source')
                if sid not in node_outputs:
                    continue
                res = node_outputs[sid]
                if res is None:
                    continue
                source_node = node_map.get(sid, {})
                source_handle = edge.get('sourceHandle')
                if source_handle and source_node.get('type') == 'ifElseNode':
                    condition_met = isinstance(res, dict) and res.get('condition_met')
                    if source_handle == 'true':
                        if condition_met is True:
                            inputs_.append(res)
                    elif source_handle == 'false':
                        if condition_met is False:
                            inputs_.append(res)
                    else:
                        inputs_.append(res)
                else:
                    inputs_.append(res)

            t0 = time.time()
            res = await process_node_async(node_, inputs_)
            t1 = time.time()
            
            duration = (t1 - t0) * 1000
            EXECUTION_STATS["nodes"][nid_]["last_execution_time_ms"] = round(duration, 2)
            EXECUTION_STATS["nodes"][nid_]["status"] = "idle"
            return nid_, res

        generation_tasks = [execute_and_time(n) for n in generation]
        
        results = await asyncio.gather(*generation_tasks)
        
        for nid, res in results:
            node_outputs[nid] = res
            
            # Update Global State for UI (Post-processing)
            # Find the node object again
            original_node = next(n for n in generation if n['id'] == nid)
            
            # [NEW] Save FULL output separately
            FULL_NODE_OUTPUTS[nid] = res # RAW OBJECT

            if original_node['type'] in ['liveDataFeed', 'liveDataFeedNode']:
                 pass
            elif original_node['type'] == 'walkForward':
                 # [DEBUG] usage
                 # print(f"DEBUG: WF {nid} updating LATEST_DATA. Res Type: {type(res)}")
                 if isinstance(res, dict):
                      # Ensure we send the full package not just 'data' so frontend can see 'fold' etc.
                      LATEST_DATA[nid] = convert_numpy(res)
            elif original_node['type'] in ['featureEngineering', 'mergeNode', 'tradingBrain', 'paperTrading', 'brokerNode', 'evalNode', 'randomAction', 'ifElseNode', 'dataStore', 'dataReplayer']:
                if isinstance(res, pd.DataFrame):
                     subset = res.tail(5).copy()
                     if 'timestamp' not in subset.columns:
                         subset['timestamp'] = subset.index.astype(str)
                     LATEST_DATA[nid] = convert_numpy(subset.to_dict('records'))
                elif isinstance(res, dict) and 'data' in res and isinstance(res['data'], pd.DataFrame):
                     # [FIX] Handle Data Package with DataFrame - PRESERVE METADATA
                     # We must convert the DataFrame to a list of records, but keep other keys (error, ts, etc.)
                     
                     new_res = res.copy()
                     subset = res['data'].tail(5).copy()
                     if 'timestamp' not in subset.columns:
                         subset['timestamp'] = subset.index.astype(str)
                     
                     new_res['data'] = subset.to_dict('records')
                     LATEST_DATA[nid] = convert_numpy(new_res)
                elif isinstance(res, dict):
                     # print(f"DEBUG: Node {nid} is dict. Keys: {res.keys()}")
                     LATEST_DATA[nid] = convert_numpy(res)
                elif isinstance(res, str) and res.startswith("Error"):
                     LATEST_DATA[nid] = {"error": res}
                elif isinstance(res, (list, tuple, int, float, np.ndarray)):
                     LATEST_DATA[nid] = convert_numpy(res)
                elif res is None:
                     # Ignore None results (waiting for data)
                     pass
                else:
                     print(f"DEBUG: Node {nid} UNKNOWN RES TYPE: {type(res)}")
            elif original_node['type'] in ['model', 'modelNode']:
                # Ensure we store a Dictionary (map of model->val) or Scalar
                # If DataFrame (standard return), take last row as Dict
                if isinstance(res, pd.DataFrame) and not res.empty:
                    LATEST_INFERENCE_RESULTS[nid] = convert_numpy(res.iloc[-1].to_dict())
                else:
                    LATEST_INFERENCE_RESULTS[nid] = convert_numpy(res)
            
    workflow_end_time = time.time()
    EXECUTION_STATS["workflow_total_time_ms"] = round((workflow_end_time - workflow_start_time) * 1000, 2)

@app.get("/stream")
def get_stream_data():
    """Returns the latest data (inputs and inference results)."""
    return convert_numpy({
        "inputs": LATEST_DATA,
        "results": LATEST_INFERENCE_RESULTS,
        "feed_snapshot": LATEST_DATA.get("FEED_SNAPSHOT", []),
        "active_feeds": LATEST_DATA.get("ACTIVE_FEEDS", []),
        "execution_stats": EXECUTION_STATS,
        "replayer_stats": REPLAY_STATS,
        "workflow_status": { "is_paused": IS_PAUSED }
    })

@app.get("/dataset")
def get_full_dataset(pair: Optional[str] = None, timeframe: Optional[str] = None):
    """Returns the full unified dataset."""
    # If specific pair/tf requested, try to get from buffer directly
    if pair and timeframe:
        key = f"{pair}_{timeframe}"
        if key in FEED_MANAGER.raw_buffers:
            df = FEED_MANAGER.raw_buffers[key].copy()
            df['timestamp'] = df.index.astype(str)
            return {"data": df.to_dict('records')}

    if FEED_MANAGER.unified_dataset.empty:
        return {"data": []}
    df = FEED_MANAGER.unified_dataset.copy()
    df['timestamp'] = df.index.astype(str)
    return {"data": df.to_dict('records')}

if __name__ == "__main__":
    import uvicorn
    
    # Initialize Multiprocessing Pool
    # Optimal workers = CPU count (default)
    PROCESS_POOL = ProcessPoolExecutor()
    
    # Start loop in background
    loop = asyncio.new_event_loop()
    threading.Thread(target=lambda: asyncio.run(data_feed_loop()), daemon=True).start()
    
    uvicorn.run(app, host="0.0.0.0", port=8001)

