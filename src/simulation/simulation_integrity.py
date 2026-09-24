# -*- coding: utf-8 -*-
"""Simulation Integrity and Leakage Audit Module"""
import pandas as pd

def audit_simulation_integrity(stage_1: pd.DataFrame, stage_2: pd.DataFrame, stage_3: pd.DataFrame) -> tuple[bool, list[str]]:
    checks = []
    reasons = []
    
    s1_keys = set(stage_1['$CASENUM'])
    s2_keys = set(stage_2['$CASENUM'])
    s3_keys = set(stage_3['$CASENUM'])
    
    overlap = s1_keys.intersection(s2_keys).intersection(s3_keys)
    checks.append(len(overlap) == 0)
    if len(overlap) > 0:
        reasons.append("Partition cross-contamination detected ($CASENUM overlap).")
        
    checks.append(stage_1['Time'].is_monotonic_increasing)
    checks.append(stage_2['Time'].is_monotonic_increasing)
    checks.append(stage_3['Time'].is_monotonic_increasing)
    if not all(checks[1:]):
        reasons.append("Chronological ordering violation within partitions.")
        
    pass_status = all(checks)
    return pass_status, reasons
