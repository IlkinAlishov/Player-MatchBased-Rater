#!/usr/bin/env python3
# train_weighted_rf.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paths ────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEAT_CSV   = os.path.join(ROOT, "data", "processed", "player_ratings_features.csv")
TARGET_CSV = os.path.join(ROOT, "data", "processed", "player_ratings_target.csv")
MODEL_PATH = os.path.join(ROOT, "models", "rating_model_weighted.pkl")

# ── Load feature matrix and target ───────────────────────────────────
X = pd.read_csv(FEAT_CSV)
y = pd.read_csv(TARGET_CSV)["rating"].values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# ── Define sample weights (tails get larger weight) ──────────────────
weights = np.where(y > 8, 3, np.where(y < 6, 2, 1))

# ── Train / test split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.2, random_state=42)

# ── Train weighted RandomForest ──────────────────────────────────────
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train, sample_weight=w_train)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(rf, MODEL_PATH)
print("Weighted RF saved →", MODEL_PATH)

# ── Evaluate ─────────────────────────────────────────────────────────
pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print(f"Weighted RF  MAE={mae:.4f}  RMSE={rmse:.4f}")

