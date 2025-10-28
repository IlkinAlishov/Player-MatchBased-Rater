#!/usr/bin/env python3
#inspect_tier_predictions.py



import os, pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── Paths ───────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEAT_CSV   = os.path.join(ROOT, "data", "processed", "player_ratings_features.csv")
TARGET_CSV = os.path.join(ROOT, "data", "processed", "player_ratings_target.csv")
CLEAN_CSV  = os.path.join(ROOT, "data", "processed", "player_ratings_cleaned.csv")
MODEL_PATH = os.path.join(ROOT, "models", "rating_model.pkl")

# ── Helper to grab ≤2 rows in each rating band ───────────────────────
def pick_examples(df_test, bands):
    picks = []
    for lo, hi in bands:
        band = df_test[(df_test["rating"] >= lo) & (df_test["rating"] < hi)]
        picks.append(band.head(2))        # take first 2 rows in band
    return pd.concat(picks, ignore_index=True)

def main():
    # 1) Load data
    X = pd.read_csv(FEAT_CSV)
    y = pd.read_csv(TARGET_CSV)["rating"].values
    cleaned = pd.read_csv(CLEAN_CSV)

    # 2) Recreate 80/20 train-test split (same seed as training)
    X_train, X_test, y_train, y_test, clean_train, clean_test = train_test_split(
        X, y, cleaned, test_size=0.2, random_state=42
    )
    # Reset indices so they align 1-to-1
    X_test = X_test.reset_index(drop=True)
    clean_test = clean_test.reset_index(drop=True)

    # 3) Build a test DataFrame with context + true rating
    test_df = clean_test.copy()
    test_df["rating"] = y_test

    # 4) Select examples in four rating tiers
    tiers = [(5.75, 6.25), (6.75, 7.25), (7.75, 8.25), (8.75, 9.25)]
    samp_df = pick_examples(test_df, tiers)
    if samp_df.empty:
        print("No examples found in the requested tiers.")
        return

    # 5) Get corresponding feature rows
    feat_rows = X_test.loc[samp_df.index]

    # 6) Load model and predict
    model = joblib.load(MODEL_PATH)
    preds = model.predict(feat_rows)

    samp_df["pred_rating"] = preds
    samp_df["abs_err"] = (samp_df["rating"] - samp_df["pred_rating"]).abs()

    # 7) Display results
    cols = ["competition", "team", "pos", "minutesPlayed",
            "rating", "pred_rating", "abs_err"]
    print("\nModel predictions on tiered test examples")
    print(samp_df[cols]
          .sort_values("rating")
          .to_string(index=False, float_format="%.2f"))

if __name__ == "__main__":
    main()
