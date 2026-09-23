# src/validation.py
import pandas as pd

def run_data_quality_gate(df: pd.DataFrame):
    """Checks raw OHLCV data for gaps, NaN values, negative prices, and sorting issues."""
    checks = {}
    df['Date'] = pd.to_datetime(df['Time'], errors='coerce')
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    checks['Missing Values'] = df[['Open', 'High', 'Low', 'Close']].isna().sum().sum() == 0
    checks['Chronological Order'] = df['Date'].is_monotonic_increasing
    checks['Zero/Negative Prices'] = (df[['Open', 'High', 'Low', 'Close']] <= 0).sum().sum() == 0
    checks['Duplicate Timestamps'] = df['Date'].duplicated().sum() == 0
    
    passed = all(checks.values())
    clean_df = df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
    return passed, checks, clean_df
