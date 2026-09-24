import pandas as pd
import numpy as np

def run_data_quality_gate(df: pd.DataFrame):
    checks = {}
    
    # 1. Missing values check
    missing_counts = df[['Open', 'High', 'Low', 'Close']].isna().sum().sum()
    checks['Missing_Values'] = bool(missing_counts == 0)
    
    # 2. Chronological ordering
    is_sorted = df['Time'].is_monotonic_increasing if 'Time' in df.columns else True
    checks['Chronological_Order'] = bool(is_sorted)
    
    # 3. Invalid OHLC check (High >= Low, High >= Open/Close, Low <= Open/Close)
    valid_ohlc = (
        (df['High'] >= df['Low']).all() and
        (df['High'] >= df['Open']).all() and
        (df['High'] >= df['Close']).all() and
        (df['Low'] <= df['Open']).all() and
        (df['Low'] <= df['Close']).all()
    )
    checks['Valid_OHLC'] = bool(valid_ohlc)
    
    # 4. Zero or negative prices
    positive_prices = (df[['Open', 'High', 'Low', 'Close']] > 0).all().all()
    checks['Positive_Prices'] = bool(positive_prices)
    
    # Overall pass status
    pass_status = all(checks.values())
    return pass_status, checks, df
