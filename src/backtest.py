# -*- coding: utf-8 -*-
"""
Walk-Forward Simulation & Backtesting Engine with Advanced Performance Metrics
"""
import pandas as pd
import numpy as np
from src.risk_engine import evaluate_tradeability_gate

def run_walk_forward_simulation(df: pd.DataFrame, features: list, models: dict, 
                                min_conviction: float = 0.55, max_disagreement: float = 30.0,
                                spread: float = 0.20, commission: float = 0.0002):
    """
    Simulates sequential trade execution and computes institutional performance metrics.
    """
    results = []
    capital = 10000.0
    equity_curve = [capital]
    trades = []
    
    start_idx = 100
    
    for i in range(start_idx, len(df) - 1):
        train_df = df.iloc[:i]
        current_row = df.iloc[[i]]
        next_row = df.iloc[i + 1]
        
        y_train = train_df['Target_Dir_1']
        valid_idx = y_train != 0.5
        if len(np.unique(y_train[valid_idx])) < 2:
            continue
            
        X_train, y_bin = train_df[features].values[valid_idx], (y_train[valid_idx] == 1).astype(int)
        
        horizon_probs = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_bin)
                p = model.predict_proba(current_row[features])[0]
                horizon_probs[name] = p[1] if len(p) > 1 else 0.5
            except Exception:
                horizon_probs[name] = 0.5
                
        probs = list(horizon_probs.values())
        p_up = np.mean(probs)
        disagreement = float(np.std(probs) * 100.0)
        uncertainty = "LOW" if disagreement < 20.0 else ("MODERATE" if disagreement < 40.0 else "HIGH")
        
        net_ev = (p_up - 0.5) * current_row['ATR'].values[0] - spread - (current_row['Close'].values[0] * commission)
        tradeability, _ = evaluate_tradeability_gate(
            p_up, net_ev, 100.0 - disagreement, True, uncertainty, 
            min_conviction=min_conviction, max_disagreement=max_disagreement
        )
        
        actual_return = next_row['Log_Return']
        trade_taken = (tradeability == "TRADEABLE")
        pnl = 0.0
        
        if trade_taken:
            direction = 1 if p_up > 0.5 else -1
            pnl = direction * actual_return * capital - (spread + commission * capital)
            capital += pnl
            trades.append(pnl)
            
        equity_curve.append(capital)
        results.append({
            "Time": current_row['Time'].values[0],
            "P_Up": p_up,
            "TradeTaken": trade_taken,
            "Capital": capital
        })
        
    sim_df = pd.DataFrame(results)
    
    # Calculate performance stats
    eq_series = pd.Series(equity_curve)
    total_return_pct = ((capital - 10000.0) / 10000.0) * 100.0
    
    # Maximum Drawdown calculation
    rolling_max = eq_series.cummax()
    drawdown = (eq_series - rolling_max) / rolling_max
    max_drawdown_pct = float(drawdown.min() * 100.0)
    
    win_rate = (sum(1 for t in trades if t > 0) / len(trades) * 100.0) if trades else 0.0
    
    metrics = {
        "Final_Equity": capital,
        "Total_Return_Pct": total_return_pct,
        "Max_Drawdown_Pct": max_drawdown_pct,
        "Win_Rate": win_rate,
        "Total_Trades": len(trades)
    }
    
    return sim_df, metrics
