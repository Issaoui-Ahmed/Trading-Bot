
from typing import Dict, Any

def decide_action(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decides the trading action based on inputs.
    
    Inputs expected:
    - prediction (from xgb_meta_model or similar): 0 (HOLD), 1 (BUY), 2 (SELL)
    - close (current price): float
    
    Output Schema (Kraken-like):
    {
        "action": "buy" | "sell" | "pass",
        "order_type": "market",
        "volume": float, (in base currency, e.g. BTC)
        "pair": str
    }
    """
    
    # 1. Parse Input
    # The input dataframe row is passed as a dict
    # extracting specific columns
    
    # 'prediction' might come from a column named 'xgb_meta_model' or just generic 'prediction'
    # We'll just look for common keys
    signal = 0
    price = 0.0
    
    for k, v in inputs.items():
        if "prediction" in k or "model" in k:
            try:
                signal = int(float(v))
            except:
                pass
        if "close" in k.lower():
            price = float(v)
            
    # Default fallback if simple lookup fails, try direct keys
    if "xgb_meta_model" in inputs:
        signal = int(float(inputs["xgb_meta_model"]))
        
    # Logic
    action = "pass"
    volume = 0.0001 # Fixed small size for testing
    
    if signal == 1:
        action = "buy"
    elif signal == 2:
        action = "sell"
        
    return {
        "action": action,
        "order_type": "market",
        "volume": volume,
        # "pair": "XBTUSD" # In future, this might be dynamic or passed from node
    }
