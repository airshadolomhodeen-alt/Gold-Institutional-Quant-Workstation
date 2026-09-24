# -*- coding: utf-8 -*-
"""
Walk-Forward Time Series Validation Framework
"""
def walk_forward_split(df, train_window=200, test_window=50):
    n = len(df)
    for i in range(train_window, n - test_window, test_window):
        yield df.iloc[i - train_window:i], df.iloc[i:i + test_window]
