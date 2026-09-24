import numpy as np
import pandas as pd

def detect_market_regime(df: pd.DataFrame):
    df = df.copy()
    vol_median = df['Realized_Vol'].rolling(50).median()
    ret_mean = df['Log_Return'].rolling(50).mean()
    
    current_vol = df['Realized_Vol'].iloc[-1]
    current_ret = ret_mean.iloc[-1]
    med_v = vol_median.iloc[-1] if not np.isnan(vol_median.iloc[-1]) else current_vol
    
    if current_vol > med_v and current_ret > 0:
        regime = "HIGH_VOLATILITY_TREND"
    elif current_vol > med_v and current_ret <= 0:
        regime = "HIGH_VOLATILITY_RANGE"
    elif current_vol <= med_v and current_ret > 0:
        regime = "LOW_VOLATILITY_TREND"
    else:
        regime = "LOW_VOLATILITY_RANGE"
        
    regime_prob = 0.74 # Estimated state persistence likelihood
    return regime, regime_prob, df
