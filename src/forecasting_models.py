# -*- coding: utf-8 -*-
"""
Forecasting Models Repository (Probabilistic Classifiers)
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def get_forecasting_models():
    """Returns a dictionary of robust classification models for multi-horizon probabilities."""
    return {
        "Logistic_Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }
