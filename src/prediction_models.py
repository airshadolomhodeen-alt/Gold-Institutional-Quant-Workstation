import numpy as np
import pandas as pd

def generate_return_predictions(df_model: pd.DataFrame, features: list, horizons: list):
    predictions = {}
    for h in horizons:
        f_ret = df_model[f'Forward_Return_{h}']
        mean_r = f_ret.mean()
        std_r = f_ret.std() if not np.isnan(f_ret.std()) else 0.01
        predictions[h] = {
            "Expected_Return": mean_r,
            "Lower_Bound": mean_r - 1.96 * std_r,
            "Upper_Bound": mean_r + 1.96 * std_r
        }
    return predictions
