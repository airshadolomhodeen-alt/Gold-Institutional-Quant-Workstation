# -*- coding: utf-8 -*-
"""
Forecasting Models Initialization
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_forecasting_models():
    return {
        "Logistic_Regression": LogisticRegression(max_iter=1000, penalty='l2', solver='lbfgs', random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    }
