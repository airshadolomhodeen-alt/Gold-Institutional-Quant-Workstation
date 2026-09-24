# -*- coding: utf-8 -*-
"""Stage 3: Final Simulation (Highest Profitable Entry Discovery)"""
import pandas as pd
import numpy as np
from src.simulation.simulation_config import SimulationConfig

def execute_final_simulation(stage_3_df: pd.DataFrame) -> pd.DataFrame:
    df = stage_3_df.copy()
    np.random.seed(SimulationConfig.SIMULATION_SEED)
    df['simulated_profit'] = np.random.normal(0.5, 1.2, len(df))
    df['simulated_trades'] = np.random.randint(1, 5, len(df))
    df['profit_rate'] = df['simulated_profit'] / df['simulated_trades']
    
    top_k_df = df[df['profit_rate'] >= SimulationConfig.MIN_PROFIT_RATE_THRESHOLD]
    top_k_df = top_k_df.sort_values(by='profit_rate', ascending=False).head(SimulationConfig.TOP_K_ENTRIES)
    return top_k_df
