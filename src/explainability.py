"""
FILE: src/explainability.py
PURPOSE: Extract true model feature importances for the explainability panel.
REASON FOR CHANGE: Replaces the equal-weight fallback (14.29%) with actual model-driven feature attribution.
"""

import pandas as pd
import numpy as np

def compute_feature_importances(model, feature_names: list) -> pd.DataFrame:
    """
    Extracts feature importances from fitted models (Tree-based or Linear).
    """
    importances = None
    
    # Check for tree-based models (Random Forest, Gradient Boosting)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # Check for linear models (Logistic Regression coefficients)
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    
    # Fallback if model hasn't been trained or doesn't support importances
    if importances is None or len(importances) != len(feature_names):
        importances = np.ones(len(feature_names)) / len(feature_names)
    
    # Normalize to percentage sum of 100%
    total = np.sum(importances)
    if total > 0:
        normalized = (importances / total) * 100
    else:
        normalized = np.ones(len(feature_names)) / len(feature_names) * 100
        
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Predictive Contribution": [f"{val:.2f}%" for val in normalized]
    })
    
    return df_imp.sort_values(by="Predictive Contribution", ascending=False)
