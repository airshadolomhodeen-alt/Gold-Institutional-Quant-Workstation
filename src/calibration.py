# src/calibration.py
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

def calibrate_probabilities(raw_probs: np.ndarray, y_true: np.ndarray):
    """Evaluates and computes calibration metrics like Brier Score and Log Loss."""
    # Simple temperature scaling / probability clipping for robust output
    clipped_probs = np.clip(raw_probs, 1e-5, 1 - 1e-5)
    brier = brier_score_loss(y_true, clipped_probs) if len(np.unique(y_true)) > 1 else 0.0
    logloss = log_loss(y_true, clipped_probs) if len(np.unique(y_true)) > 1 else 0.0
    return clipped_probs, {"Brier Score": brier, "Log Loss": logloss}
