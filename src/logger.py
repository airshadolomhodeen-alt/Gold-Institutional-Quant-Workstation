# -*- coding: utf-8 -*-
"""
Prediction & Performance Tracking Logger
"""
import os
import pandas as pd
from datetime import datetime

LOG_FILE = "logs/prediction_audit_trail.csv"

def initialize_logger():
    """Ensures the logs directory and audit trail file exist."""
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        df_init = pd.DataFrame(columns=[
            "Timestamp", "Symbol", "Interval", "Current_Price", 
            "Horizon", "P_Up", "P_Down", "Expected_Return", 
            "Lower_Bound_95", "Upper_Bound_95", "Actual_Price_At_Horizon"
        ])
        df_init.to_csv(LOG_FILE, index=False)

def log_current_predictions(symbol, interval, current_price, horizons, prob_up_list, prob_down_list, return_predictions):
    """Logs the latest 10-horizon forecast state for post-hoc evaluation."""
    initialize_logger()
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    new_rows = []
    
    for idx, h in enumerate(horizons):
        p_up = prob_up_list[idx] if idx < len(prob_up_list) else 0.0
        p_down = prob_down_list[idx] if idx < len(prob_down_list) else 0.0
        reg_data = return_predictions.get(h, {"Expected_Return": 0.0, "Lower_Bound": 0.0, "Upper_Bound": 0.0})
        
        new_rows.append({
            "Timestamp": timestamp,
            "Symbol": symbol,
            "Interval": interval,
            "Current_Price": current_price,
            "Horizon": f"T+{h}",
            "P_Up": round(p_up, 4),
            "P_Down": round(p_down, 4),
            "Expected_Return": round(reg_data["Expected_Return"], 5),
            "Lower_Bound_95": round(reg_data["Lower_Bound"], 5),
            "Upper_Bound_95": round(reg_data["Upper_Bound"], 5),
            "Actual_Price_At_Horizon": None # Filled later when realized
        })
        
    df_logs = pd.read_csv(LOG_FILE)
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_logs, df_new], ignore_index=True)
    df_combined.to_csv(LOG_FILE, index=False)
