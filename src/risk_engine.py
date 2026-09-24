# -*- coding: utf-8 -*-
"""
Tradeability Gate Risk Engine
"""
def evaluate_tradeability_gate(p_up: float, net_ev: float, model_agreement: float, dq_pass: bool, uncertainty: str):
    reasons = []
    if not dq_pass:
        reasons.append("Data Quality Failure")
    if net_ev <= 0:
        reasons.append("Negative Net Expected Value after Costs")
    if model_agreement < 60.0:
        reasons.append("Insufficient Model Consensus")
    if uncertainty == "HIGH":
        reasons.append("Elevated Forecast Uncertainty")
        
    if reasons:
        return "NO-TRADE", reasons
    return "TRADEABLE", []
