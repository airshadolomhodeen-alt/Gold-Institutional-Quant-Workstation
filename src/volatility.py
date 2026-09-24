import numpy as np
import pandas as pd

def compute_volatility_features(df: pd.DataFrame, window: int = 20, lam: float = 0.94) -> pd.DataFrame:
    df = df.copy()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Realized Volatility (Annualized assuming intraday/daily context)
    df['Realized_Vol'] = df['Log_Return'].rolling(window=window).std() * np.sqrt(252)
    
    # EWMA Volatility
    ewma_var = pd.Series(0.0, index=df.index)
    var_t = df['Log_Return'].var()
    for i in range(1, len(df)):
        r_prev = df['Log_Return'].iloc[i-1]
        var_t = lam * var_t + (1 - lam) * (r_prev ** 2) if not np.isnan(r_prev) else var_t
        ewma_var.iloc[i] = var_t
    df['EWMA_Vol'] = np.sqrt(ewma_var) * np.sqrt(252)
    return df
