# -*- coding: utf-8 -*-
"""
Expected Value & Transaction Cost Engine
"""
def compute_expected_values(p_up: float, p_down: float, exp_return: float, spread: float, commission: float, slippage: float, atr: float):
    gross_ev = exp_return * 100.0
    total_costs = spread + (commission * 2000.0) + slippage
    net_ev = gross_ev - total_costs
    rr_ratio = 1.2 if p_up > p_down else 0.8
    
    return {
        "Gross_EV": gross_ev,
        "Total_Costs": total_costs,
        "Net_EV": net_ev,
        "RR_Ratio": rr_ratio
    }
