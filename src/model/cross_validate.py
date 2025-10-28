#!/usr/bin/env python3
#cross_validate.py


import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEAT_CSV     = os.path.join(PROJECT_ROOT, "data", "processed", "player_ratings_features.csv")
TARGET_CSV   = os.path.join(PROJECT_ROOT, "data", "processed", "player_ratings_target.csv")

# custom RMSE scorer
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
rmse_scorer = make_scorer(rmse, greater_is_better=False)  # negative for sklearn


def main():
    # 1) Load engineered features and target
    X = pd.read_csv(FEAT_CSV)
    y = pd.read_csv(TARGET_CSV)["rating"].values

    # Handle any remaining infinities / NaNs
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    # 2) Define model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # 3) 5‑fold CV
    mae_scores = cross_val_score(rf, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    rmse_scores = cross_val_score(rf, X, y, cv=5, scoring=rmse_scorer, n_jobs=-1)

    mae_mean, mae_std = -mae_scores.mean(), mae_scores.std()
    rmse_mean, rmse_std = -rmse_scores.mean(), rmse_scores.std()

    print(f"5‑fold CV MAE :  {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"5‑fold CV RMSE:  {rmse_mean:.4f} ± {rmse_std:.4f}")

if __name__ == "__main__":
    main()
