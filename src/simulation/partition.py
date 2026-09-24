# -*- coding: utf-8 -*-
"""Chronological Partition Enforcement (60 / 20 / 20)"""
import pandas as pd
from typing import Tuple

def enforce_chronological_partition(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n1 = int(n * 0.60)
    n2 = int(n * 0.80)
    
    stage_1 = df.iloc[:n1].copy()
    stage_1['partition'] = "PRE_SIM_1"
    
    stage_2 = df.iloc[n1:n2].copy()
    stage_2['partition'] = "PRE_SIM_2"
    
    stage_3 = df.iloc[n2:].copy()
    stage_3['partition'] = "FINAL_SIM"
    
    return stage_1, stage_2, stage_3
