import numpy as np
from sklearn.isotonic import IsotonicRegression

def calibrate_probabilities(raw_probs: np.ndarray, true_labels: np.ndarray):
    ir = IsotonicRegression(out_of_bounds='clip')
    try:
        if len(np.unique(true_labels)) > 1:
            ir.fit(raw_probs, true_labels)
            calibrated = ir.predict(raw_probs)
        else:
            calibrated = raw_probs
    except Exception:
        calibrated = raw_probs
    brier = np.mean((calibrated - true_labels) ** 2)
    log_loss = -np.mean(true_labels * np.log(np.clip(calibrated, 1e-15, 1-1e-15)) + (1 - true_labels) * np.log(np.clip(1-calibrated, 1e-15, 1-1e-15)))
    return calibrated, brier, log_loss
