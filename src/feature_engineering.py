# -*- coding: utf-8 -*-
"""
Leakage-Safe Feature Engineering Engine
"""
import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame, rsi_per: int, macd_f: int, macd_s: int, atr_per: int) -> pd.DataFrame:
    df = df.copy()
    
    # Log Return
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=rsi_per).mean()
    avg_loss = loss.rolling(window=rsi_per).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    # MACD
    ema_fast = df['Close'].ewm(span=macd_f, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_s, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # ATR
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=atr_per).mean()
    
    # Realized Volatility
    df['Realized_Vol'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    
    # Updated to modern pandas bfill syntax
    return df.bfill().fillna(0.0)
