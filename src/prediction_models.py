# -*- coding: utf-8 -*-
"""
Multi-Horizon Predictive Regression Engine (Magnitude & Bounds)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

def get_regression_models():
    """Returns a dictionary of robust regression models for point-prediction."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "GB_Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    }

def generate_return_predictions(df: pd.DataFrame, features: list, horizons: list = list(range(1, 11))):
    """
    Trains regression models for each horizon to predict exact forward return magnitudes
    and estimate predictive intervals.
    """
    reg_models = get_regression_models()
    predictions = {}
    
    X = df[features].values
    X_latest = df[features].iloc[[-1]].values
    
    for h in horizons:
        target_col = f'Forward_Return_{h}'
        if target_col not in df.columns:
            continue
            
        y = df[target_col].values
        valid = ~np.isnan(y)
        X_train, y_train = X[valid], y[valid]
        
        horizon_preds = []
        for name, model in reg_models.items():
            try:
                model.fit(X_train, y_train)
                pred = model.predict(X_latest)[0]
                horizon_preds.append(pred)
            except Exception:
                horizon_preds.append(0.0)
                
        mean_pred = np.mean(horizon_preds) if horizon_preds else 0.0
        std_pred = np.std(horizon_preds) if len(horizon_preds) > 1 else 0.001
        
        predictions[h] = {
            "Expected_Return": mean_pred,
            "Lower_Bound": mean_pred - 1.96 * std_pred,
            "Upper_Bound": mean_pred + 1.96 * std_pred
        }
        
    return predictions
