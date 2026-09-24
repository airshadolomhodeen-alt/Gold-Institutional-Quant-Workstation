# -*- coding: utf-8 -*-
"""
Multi-Horizon Targets Module (Classification & Regression)
"""
import pandas as pd
import numpy as np

def create_multi_horizon_targets(df: pd.DataFrame, threshold_type: str = 'atr', threshold_multiplier: float = 0.5):
    """
    Creates multi-horizon directional classification targets and continuous regression return targets 
    for horizons h = 1 through 10.
    """
    for h in range(1, 11):
        # Continuous regression target: Forward log return
        df[f'Forward_Return_{h}'] = np.log(df['Close'].shift(-h) / df['Close'])
        
        # Dynamic classification threshold
        if threshold_type == 'atr' and 'ATR' in df.columns:
            threshold = threshold_multiplier * (df['ATR'] / df['Close'])
        else:
            threshold = 0.001 * h
            
        # Discrete classification labels: 1 = UP, 0 = DOWN, 0.5 = NEUTRAL
        cond_up = df[f'Forward_Return_{h}'] > threshold
        cond_down = df[f'Forward_Return_{h}'] < -threshold
        
        target = np.full(len(df), 0.5)
        target[cond_up] = 1.0
        target[cond_down] = 0.0
        df[f'Target_Dir_{h}'] = target
        
    return df
