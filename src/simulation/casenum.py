# -*- coding: utf-8 -*-
"""CASENUM Surrogate Key Allocation Module"""
import pandas as pd

def assign_casenum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['$CASENUM'] = [f"CASENUM_{i+1:05d}" for i in range(len(df))]
    return df
