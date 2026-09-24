from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def get_forecasting_models():
    return {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
