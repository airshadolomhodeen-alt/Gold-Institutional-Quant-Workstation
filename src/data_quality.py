# -*- coding: utf-8 -*-
"""
Automated Data Quality Gate
"""
import pandas as pd
import numpy as np

def run_data_quality_gate(df: pd.DataFrame):
    checks = {}
    passed = True
    
    required_cols = ['Time', 'Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            checks[f"Missing Column: {col}"] = False
            passed = False
        else:
            checks[f"Missing Column: {col}"] = True

    if not passed:
        return False, checks, df

    # Check NaNs
    nan_count = df[required_cols].isna().sum().sum()
    checks["No Missing OHLC Values"] = nan_count == 0
    if nan_count > 0:
        passed = False
        df = df.dropna(subset=required_cols)

    # Check zero/negative prices
    valid_prices = (df[['Open', 'High', 'Low', 'Close']] > 0).all().all()
    checks["Positive Prices"] = valid_prices
    if not valid_prices:
        passed = False

    # Check duplicate timestamps
    dups = df['Time'].duplicated().sum()
    checks["No Duplicate Timestamps"] = dups == 0
    if dups > 0:
        df = df.drop_duplicates(subset=['Time']).reset_index(drop=True)

    return passed, checks, df
