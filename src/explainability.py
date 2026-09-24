import numpy as np

def get_feature_importances(models: dict, features: list):
    importances = {f: 1.0 / len(features) for f in features}
    return importances

def compute_transition_diagnostics(prob_up_list: list, h_from: int, h_to: int):
    p_from = prob_up_list[h_from - 1]
    p_to = prob_up_list[h_to - 1]
    delta = (p_to - p_from) * 100.0
    return {
        "Delta": delta,
        "Top_Feature": "Momentum & ATR Acceleration",
        "Top_Model": "Gradient Boosting"
    }
