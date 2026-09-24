# -*- coding: utf-8 -*-
"""
Volatility Modeling Engine (EWMA & GARCH Proxy)
"""
import pandas as pd
import numpy as np

def compute_volatility_features(df: pd.DataFrame, lambda_ewma=0.94) -> pd.DataFrame:
    df = df.copy()
    returns_sq = df['Log_Return'] ** 2
    
    ewma_var = [returns_sq.iloc[0]]
    for r2 in returns_sq.iloc[1:]:
        ewma_var.append(lambda_ewma * ewma_var[-1] + (1 - lambda_ewma) * r2)
        
    df['EWMA_Vol'] = np.sqrt(ewma_var) * np.sqrt(252)
    return df
