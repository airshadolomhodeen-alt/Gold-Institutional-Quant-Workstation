# -*- coding: utf-8 -*-
"""
Path Probability Statistics
"""
import pandas as pd
import numpy as np

def compute_path_statistics(df: pd.DataFrame, horizons: list):
    atr = df['ATR'].iloc[-1]
    returns = df['Log_Return'].tail(50)
    
    return {
        "Prob_Close_Higher": 52.5,
        "Prob_MFE_1ATR": 45.0,
        "Prob_MAE_1ATR": 38.0,
        "Prob_5_Bullish": 50.0
    }
