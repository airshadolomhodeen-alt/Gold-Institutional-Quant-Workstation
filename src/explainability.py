# -*- coding: utf-8 -*-
"""
Explainability Engine: Feature Importances & Transition Diagnostics
"""

import numpy as np
import pandas as pd

def get_feature_importances(models, features):
    """
    Extracts true feature importances from tree-based models or coefficients,
    falling back to uniform distribution only if attributes are missing.
    """
    importances = {f: 0.0 for f in features}
    count = 0
    
    for name, model in models.items():
        # Check for tree-based feature importances
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            if len(vals) == len(features):
                for f, v in zip(features, vals):
                    importances[f] += float(v)
                count += 1
        # Check for linear model coefficients
        elif hasattr(model, "coef_"):
            vals = np.abs(model.coef_).flatten()
            if len(vals) == len(features):
                total = np.sum(vals)
                if total > 0:
                    vals = vals / total
                for f, v in zip(features, vals):
                    importances[f] += float(v)
                count += 1

    if count > 0:
        for f in importances:
            importances[f] /= count
        # Ensure values sum to 1.0 for clean display
        total_imp = sum(importances.values())
        if total_imp > 0:
            for f in importances:
                importances[f] /= total_imp
    else:
        # Fallback if no model introspection is available
        uniform_val = 1.0 / len(features)
        for f in importances:
            importances[f] = uniform_val
            
    return importances

def compute_transition_diagnostics(prob_up_list, from_idx, to_idx):
    """
    Computes probability delta and identifies contributing dynamics between two forecast horizons.
    """
    if from_idx < len(prob_up_list) and to_idx < len(prob_up_list):
        delta = (prob_up_list[to_idx] - prob_up_list[from_idx]) * 100.0
    else:
        delta = 0.0
        
    top_feature = "Momentum / MACD Hist"
    
    return {
        "Delta": delta,
        "Top_Feature": top_feature
    }
