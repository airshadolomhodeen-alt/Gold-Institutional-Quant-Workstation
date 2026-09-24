# -*- coding: utf-8 -*-
"""
Multi-Horizon Targets Module - Asymmetric Trend Weighting & Forward Returns
"""

import numpy as np
import pandas as pd

def create_multi_horizon_targets(df, threshold_type='atr', threshold_multiplier=0.5):
    """
    Creates multi-horizon directional targets and stores forward returns 
    to prevent KeyErrors in prediction models.
    """
    df = df.copy()
    horizons = list(range(1, 11))
    
    for h in horizons:
        # Future return over horizon h
        future_return = df['Close'].shift(-h) - df['Close']
        
        # Explicitly save forward return for prediction models
        df[f'Forward_Return_{h}'] = future_return
        
        if threshold_type == 'atr':
            threshold = df['ATR'] * threshold_multiplier * np.sqrt(h)
        else:
            threshold = df['Close'] * 0.001 * h
            
        # Asymmetric adjustment based on 14-period linear regression slope
        trend_bias = np.where(df['LR_Slope_14'] < 0, -0.2 * threshold, 0.2 * threshold)
        adjusted_threshold_up = threshold - trend_bias
        adjusted_threshold_down = threshold + trend_bias

        # Default class: 0.5 (Neutral)
        target = np.full(len(df), 0.5)
        
        # UP = 1.0, DOWN = 0.0
        target[future_return > adjusted_threshold_up] = 1.0
        target[future_return < -adjusted_threshold_down] = 0.0
        
        df[f'Target_Dir_{h}'] = target
        
    return df
