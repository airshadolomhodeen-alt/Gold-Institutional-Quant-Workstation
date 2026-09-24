# -*- coding: utf-8 -*-
"""
Model Ensemble & Disagreement Engine
"""
import numpy as np

def aggregate_ensemble(horizon_probs: dict):
    probs = list(horizon_probs.values())
    ensemble_prob = np.mean(probs)
    disagreement = np.std(probs)
    return ensemble_prob, disagreement
