def compute_expected_values(p_up: float, p_down: float, expected_return: float, spread: float, commission: float, slippage: float, atr: float):
    total_costs = spread + commission + slippage
    gross_ev = expected_return * 100.0
    net_ev = gross_ev - total_costs
    rr_ratio = 1.5 if p_up > p_down else 1.2
    return {
        "Gross_EV": gross_ev,
        "Total_Costs": total_costs,
        "Net_EV": net_ev,
        "RR_Ratio": rr_ratio
    }
