# src/walk_forward.py
import pandas as pd

def walk_forward_split(df: pd.DataFrame, train_window: int = 500, test_window: int = 50):
    """Generator for rolling walk-forward cross-validation splits without look-ahead bias."""
    total_len = len(df)
    start_idx = 0
    while start_idx + train_window + test_window <= total_len:
        train_end = start_idx + train_window
        test_end = train_end + test_window
        yield start_idx, train_end, test_end
        start_idx += test_window
