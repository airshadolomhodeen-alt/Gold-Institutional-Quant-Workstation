import numpy as np

def aggregate_ensemble(horizon_probs: dict):
    probs = list(horizon_probs.values())
    if not probs:
        return 0.5, 0.0
    p_ens = float(np.mean(probs))
    disagreement = float(np.std(probs))
    return p_ens, disagreement
