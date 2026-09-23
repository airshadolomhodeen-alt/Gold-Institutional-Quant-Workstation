# src/feature_engineering.py
import numpy as np
import pandas as pd

def compute_features(df: pd.DataFrame, rsi_per: int = 14, macd_f: int = 12, macd_s: int = 26, atr_per: int = 14):
    """Computes technical indicators, log returns, and volatility measures using past data only."""
    df = df.copy()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Return'] = df['Close'].pct_change()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_per).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_per).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    # MACD
    exp1 = df['Close'].ewm(span=macd_f, adjust=False).mean()
    exp2 = df['Close'].ewm(span=macd_s, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # ATR & Volatility
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(atr_per).mean()
    df['Realized_Vol'] = df['Log_Return'].rolling(20).std() * np.sqrt(252)
    
    return df
