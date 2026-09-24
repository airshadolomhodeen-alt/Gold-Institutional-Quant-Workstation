# -*- coding: utf-8 -*-
"""
Model Ensemble & Disagreement Engine
"""
import numpy as np

def aggregate_ensemble(horizon_probs: dict):
    """
    Aggregates individual model probability dictionary into an ensemble mean and disagreement score.
    """
    probs = list(horizon_probs.values())
    ensemble_prob = np.mean(probs)
    disagreement = np.std(probs)
    return ensemble_prob, disagreement

def compute_model_disagreement(horizon_probs: dict):
    """
    Computes the dispersion (standard deviation) across individual model probability outputs.
    """
    probs = list(horizon_probs.values())
    if not probs:
        return 0.0
    return float(np.std(probs))
