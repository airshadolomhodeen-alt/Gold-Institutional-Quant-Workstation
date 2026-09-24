# -*- coding: utf-8 -*-
"""
Multi-Horizon Target Generator
"""
import pandas as pd
import numpy as np

def create_multi_horizon_targets(df: pd.DataFrame, threshold_type='atr', threshold_multiplier=0.5):
    df = df.copy()
    for h in range(1, 11):
        fwd_return = np.log(df['Close'].shift(-h) / df['Close'])
        df[f'Forward_Return_{h}'] = fwd_return
        
        if threshold_type == 'atr':
            threshold = (df['ATR'] / df['Close']) * threshold_multiplier
        else:
            threshold = 0.001
            
        conditions = [
            fwd_return > threshold,
            fwd_return < -threshold
        ]
        choices = [1, -1]
        df[f'Target_Dir_{h}'] = np.select(conditions, choices, default=0)
        
    return df
