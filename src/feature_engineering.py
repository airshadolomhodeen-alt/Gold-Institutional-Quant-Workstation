import pandas as pd
from src.indicators import compute_rsi, compute_macd, compute_atr

def compute_features(df: pd.DataFrame, rsi_period: int, macd_fast: int, macd_slow: int, atr_period: int) -> pd.DataFrame:
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], period=rsi_period)
    df['MACD'], df['Signal'], df['MACD_Hist'] = compute_macd(df['Close'], fast=macd_fast, slow=macd_slow)
    df['ATR'] = compute_atr(df, period=atr_period)
    df['Candle_Body'] = df['Close'] - df['Open']
    df['Candle_Range'] = df['High'] - df['Low']
    df['Body_Pct'] = df['Candle_Body'] / (df['Candle_Range'] + 1e-10)
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    return df
