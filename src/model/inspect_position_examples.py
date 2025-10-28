#!/usr/bin/env python3
# inspect_position_examples.py


import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# ── Project‑relative paths ───────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")
FEAT_CSV   = os.path.join(PROC, "player_ratings_features.csv")
TARGET_CSV = os.path.join(PROC, "player_ratings_target.csv")
CLEAN_CSV  = os.path.join(PROC, "player_ratings_cleaned.csv")
BASEMODEL  = os.path.join(ROOT, "models", "rating_model.pkl")
MODELDIR   = os.path.join(ROOT, "models")

# ── Load data ────────────────────────────────────────────────────────
X = pd.read_csv(FEAT_CSV)
y = pd.read_csv(TARGET_CSV)["rating"].values
clean = pd.read_csv(CLEAN_CSV)  # has pos, team, competition, minutes

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Recreate the same 80/20 split as training
X_tr, X_te, y_tr, y_te, clean_tr, clean_te = train_test_split(
    X, y, clean, test_size=0.2, random_state=42)
X_te = X_te.reset_index(drop=True)
clean_te = clean_te.reset_index(drop=True)

# ── Helper: load pos‑specific model or fallback ─────────────────────
def model_for_pos(pos):
    f = os.path.join(MODELDIR, f"rf_pos_{pos}.pkl")
    return joblib.load(f) if os.path.exists(f) else joblib.load(BASEMODEL)

# rating bands we want to sample
BANDS = [ (5.75, 6.25), (6.75, 7.25), (7.75, 10.01) ]

rows = []
for pos in sorted(clean_te["pos"].unique()):
    mask_pos = clean_te["pos"] == pos
    df_pos   = clean_te[mask_pos]
    X_pos    = X_te[mask_pos]
    if df_pos.empty:
        continue

    model = model_for_pos(pos)

    for lo, hi in BANDS:
        band_rows = df_pos[(df_pos["rating"] >= lo) & (df_pos["rating"] < hi)]
        if band_rows.empty:
            continue
        row = band_rows.iloc[0]  # pick first
        pred = model.predict(X_pos.loc[[row.name]])[0]
        rows.append({
            "pos": pos,
            "competition": row["competition"],
            "team": row["team"],
            "minutes": int(row["minutesPlayed"]),
            "true": row["rating"],
            "pred": pred,
            "abs_err": abs(row["rating"] - pred)
        })

# ── Display ─────────────────────────────────────────────────────────
print("\nPosition‑specific model sample predictions")
print(f"{'Pos':3}  {'Competition':18} {'Team':18}  Min  True  Pred  AbsErr")
for r in rows:
    print(f"{r['pos']:3}  {r['competition'][:18]:18} {r['team'][:18]:18}  {r['minutes']:3d}  "
          f"{r['true']:.2f}  {r['pred']:.2f}   {r['abs_err']:.2f}")
