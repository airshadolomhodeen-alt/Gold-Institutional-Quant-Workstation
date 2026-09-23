# src/targets.py
import numpy as np
import pandas as pd

def create_multi_horizon_targets(df: pd.DataFrame, classification_threshold: float = 0.001):
    """Generates forward returns and discrete directional targets for horizons h = 1 through 10."""
    df = df.copy()
    for h in range(1, 11):
        df[f'Forward_Return_{h}'] = np.log(df['Close'].shift(-h) / df['Close'])
        df[f'Z_Return_{h}'] = df[f'Forward_Return_{h}'] / df['ATR']
        df[f'Target_Dir_{h}'] = np.where(df[f'Forward_Return_{h}'] > classification_threshold, 1,
                                 np.where(df[f'Forward_Return_{h}'] < -classification_threshold, 0, 0.5))
    return df
