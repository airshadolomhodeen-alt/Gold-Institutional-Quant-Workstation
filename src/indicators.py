import numpy as np
import pandas as pd

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_true_range(df: pd.DataFrame) -> pd.Series:
    hl = df['High'] - df['Low']
    hc = np.abs(df['High'] - df['Close'].shift(1))
    lc = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = compute_true_range(df)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
