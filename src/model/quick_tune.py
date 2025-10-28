#!/usr/bin/env python3
# quick_tune.py


import os
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# ── Paths ─────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEAT_CSV   = os.path.join(ROOT, "data", "processed", "player_ratings_features.csv")
TARGET_CSV = os.path.join(ROOT, "data", "processed", "player_ratings_target.csv")

# ── Load data ─────────────────────────────────────────────────────────
X = pd.read_csv(FEAT_CSV)
y = pd.read_csv(TARGET_CSV)["rating"].values

# Clean infinities / NaNs just in case
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# ── Define model & parameter distributions ───────────────────────────
rf = RandomForestRegressor(random_state=42)
param_dist = {
    "n_estimators": randint(100, 401),      # 100‑400
    "max_depth":    [None, 10, 20, 40],
    "min_samples_leaf": randint(1, 5)      # 1‑4
}

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,                  # 20 random combos → quick
    cv=3,
    scoring="neg_mean_absolute_error",
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("Running RandomizedSearchCV — this should finish in a few minutes…")
search.fit(X, y)

best_mae = -search.best_score_
print("\nBest MAE : {:.4f}".format(best_mae))
print("Best params:", search.best_params_)
