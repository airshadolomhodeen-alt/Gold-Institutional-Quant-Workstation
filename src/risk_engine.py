def evaluate_tradeability_gate(p_up: float, net_ev: float, model_agreement: float, dq_pass: bool, uncertainty: str, min_conviction: float = 0.60, max_disagreement: float = 25.0):
    reasons = []
    if not dq_pass:
        return "DATA QUALITY FAILURE", ["Data quality gate failed."]
    if uncertainty == "HIGH":
        reasons.append("Predictive uncertainty elevated.")
    if (100.0 - model_agreement) > max_disagreement:
        reasons.append("Model disagreement exceeds maximum tolerance.")
    if net_ev <= 0:
        reasons.append("Expected edge insufficient after transaction costs.")
    if max(p_up, 1.0 - p_up) < min_conviction:
        reasons.append("Probability conviction below minimum threshold.")
        
    if reasons:
        return "NO-TRADE", reasons
    return "TRADEABLE", []
