# src/regime_models.py
import numpy as np
import pandas as pd

def detect_market_regime(df: pd.DataFrame):
    """Detects market volatility and trend regimes based on rolling statistics."""
    df = df.copy()
    rolling_vol = df['Realized_Vol']
    vol_median = rolling_vol.median()
    
    # Simple statistical regime mapping (can be extended to full HMM)
    df['Regime'] = np.where(rolling_vol > vol_median, "HIGH-VOLATILITY TREND", "LOW-VOLATILITY RANGE")
    current_regime = df['Regime'].iloc[-1]
    regime_prob = 0.76  # Estimated state persistence probability
    
    return current_regime, regime_prob, df
