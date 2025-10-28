#!/usr/bin/env python3
# train_pos_models.py

import os, joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEAT_CSV   = os.path.join(ROOT, "data", "processed", "player_ratings_features.csv")
TARGET_CSV = os.path.join(ROOT, "data", "processed", "player_ratings_target.csv")
CLEAN_CSV  = os.path.join(ROOT, "data", "processed", "player_ratings_cleaned.csv")
MODELDIR   = os.path.join(ROOT, "models")
os.makedirs(MODELDIR, exist_ok=True)

X = pd.read_csv(FEAT_CSV)
y = pd.read_csv(TARGET_CSV)["rating"].values
pos = pd.read_csv(CLEAN_CSV)["pos"].values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

for p in np.unique(pos):
    idx = np.where(pos == p)[0]
    X_p, y_p = X.iloc[idx], y[idx]
    X_tr, X_te, y_tr, y_te = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_tr, y_tr)
    joblib.dump(rf, os.path.join(MODELDIR, f"rf_pos_{p}.pkl"))
    mae = mean_absolute_error(y_te, rf.predict(X_te))
    print(f"{p}: MAE = {mae:.4f}  (rows={len(y_p)})")
