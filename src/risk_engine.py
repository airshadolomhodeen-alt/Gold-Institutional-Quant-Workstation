# src/risk_engine.py
def evaluate_trading_decision(max_prob: float, neutral_band: float, avg_win: float, 
                              avg_loss: float, spread_ticks: float, commission_cost: float, dq_pass: bool):
    """Calculates Expected Value after friction and triggers NO-TRADE if edge is insufficient."""
    ev = (max_prob * avg_win) - ((1 - max_prob) * avg_loss) - spread_ticks - commission_cost
    
    if max_prob >= neutral_band and ev > 0 and dq_pass:
        return "TRADEABLE", ev
    else:
        return "NO-TRADE / INSUFFICIENT EDGE", ev
