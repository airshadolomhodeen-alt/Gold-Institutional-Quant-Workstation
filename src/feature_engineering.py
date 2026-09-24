import numpy as np
import pandas as pd

def compute_features(df, rsi_period=14, macd_fast=12, macd_slow=26, atr_period=14):
    """
    Computes technical features including trend slopes and momentum metrics.
    """
    # Base price changes
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Hist'] = df['MACD'] - df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR & Volatility
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=atr_period).mean()
    df['Realized_Vol'] = df['Log_Return'].rolling(window=20).std()
    df['EWMA_Vol'] = df['Log_Return'].ewm(span=20).std()

    # ==========================================
    # TREND & SLOPE FEATURES (PART 42 ADDITION)
    # ==========================================
    # 10-candle and 20-candle moving average slopes
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_10_Slope'] = df['SMA_10'].diff(3) / 3  # Rate of change over 3 periods
    df['SMA_20_Slope'] = df['SMA_20'].diff(5) / 5
    
    # Linear regression slope over a 14-candle rolling window
    def rolling_slope(array):
        x = np.arange(len(array))
        if len(array) < 2 or np.isnan(array).any():
            return 0.0
        slope, _ = np.polyfit(x, array, 1)
        return slope

    df['LR_Slope_14'] = df['Close'].rolling(window=14).apply(rolling_slope, raw=True)
    
    return df.fillna(method='bfill').fillna(0)
