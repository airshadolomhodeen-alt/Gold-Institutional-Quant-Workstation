# src/forecasting_models.py
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_forecasting_model(model_type: str):
    """Factory function for selecting the active quant model."""
    if model_type == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    elif model_type == "Random Forest":
        return RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    elif model_type == "L2 Regularized Logistic":
        return LogisticRegression(C=0.1, random_state=42)
    else:
        return LogisticRegression(C=1.0, random_state=42)
