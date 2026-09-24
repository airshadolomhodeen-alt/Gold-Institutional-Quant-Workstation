# -*- coding: utf-8 -*-
"""
Market Regime Detection Engine
"""
import pandas as pd
import numpy as np

def detect_market_regime(df: pd.DataFrame):
    if len(df) < 50:
        return "LOW_VOLATILITY_RANGE", 0.5, df
        
    vol_median = df['ATR'].rolling(50).median().iloc[-1]
    current_atr = df['ATR'].iloc[-1]
    
    trend_metric = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
    
    if current_atr > vol_median * 1.2:
        regime = "HIGH_VOLATILITY_TREND" if abs(trend_metric) > 0.02 else "HIGH_VOLATILITY_RANGE"
    else:
        regime = "LOW_VOLATILITY_TREND" if abs(trend_metric) > 0.01 else "LOW_VOLATILITY_RANGE"
        
    df['Market_Regime'] = regime
    return regime, 0.75, df
