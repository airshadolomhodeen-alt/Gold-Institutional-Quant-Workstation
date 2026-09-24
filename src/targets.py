import numpy as np
import pandas as pd

def create_multi_horizon_targets(df: pd.DataFrame, horizons=list(range(1, 11)), threshold_multiplier: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    for h in horizons:
        df[f'Forward_Return_{h}'] = np.log(df['Close'].shift(-h) / df['Close'])
        
        # ATR-normalized threshold
        threshold = df['ATR'] * threshold_multiplier / df['Close']
        
        conditions = [
            df[f'Forward_Return_{h}'] > threshold,
            df[f'Forward_Return_{h}'] < -threshold
        ]
        choices = [1, -1] # 1: UP, -1: DOWN, 0.5: NEUTRAL
        df[f'Target_Dir_{h}'] = np.select(conditions, choices, default=0.5)
    return df
