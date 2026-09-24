# -*- coding: utf-8 -*-
"""
Risk Engine & Tradeability Gate
"""

def evaluate_tradeability_gate(prob_up, net_ev, model_agreement_pct, dq_pass, uncertainty, 
                               min_conviction=0.55, max_disagreement=30.0):
    """
    Evaluates whether market conditions meet institutional risk and edge criteria.
    """
    reasons = []
    
    if not dq_pass:
        reasons.append("Data Quality Gate Failed")
        
    if net_ev <= 0:
        reasons.append("Negative Net Expected Value after Costs")
        
    max_prob = max(prob_up, 1.0 - prob_up)
    if max_prob < min_conviction:
        reasons.append(f"Conviction ({max_prob:.1%}) below minimum threshold ({min_conviction:.1%})")
        
    if (100.0 - model_agreement_pct) > max_disagreement:
        reasons.append("Model Disagreement dispersion too high")
        
    if uncertainty == "HIGH":
        reasons.append("Ensemble uncertainty level is HIGH")
        
    if reasons:
        return "NO-TRADE", reasons
    else:
        return "TRADEABLE", []
