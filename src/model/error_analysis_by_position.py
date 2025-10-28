#!/usr/bin/env python3
##error_analysis_by_position.py


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLEAN_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "player_ratings_cleaned.csv")
FEAT_CSV     = os.path.join(PROJECT_ROOT, "data", "processed", "player_ratings_features.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "rating_model.pkl")
REPORT_PATH  = os.path.join(PROJECT_ROOT, "reports", "residuals_by_position.png")


def main():
    # 1) Load cleaned data (for pos + rating)
    cleaned_df = pd.read_csv(CLEAN_CSV)
    if "pos" not in cleaned_df.columns or "rating" not in cleaned_df.columns:
        raise ValueError("'pos' or 'rating' column missing in cleaned cleaned_df")

    # 2) Load engineered feature matrix (matches the model)
    X_all = pd.read_csv(FEAT_CSV)
    y_all = cleaned_df["rating"].values  # same ordering because features built from this
    pos_all = cleaned_df["pos"].values

    # 3) Train/test split with same seed
    X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
        X_all, y_all, pos_all, test_size=0.2, random_state=42
    )

    # 4) Load trained model
    model = joblib.load(MODEL_PATH)

    # 5) Predict on test
    y_pred = model.predict(X_test)

    # 6) Compute MAE per position
    df_res = pd.DataFrame({
        "pos": pos_test,
        "y_true": y_test,
        "y_pred": y_pred
    })
    print("\nMAE by player position:")
    for p, grp in df_res.groupby("pos"):
        mae = mean_absolute_error(grp["y_true"], grp["y_pred"])
        print(f"  {p}: {mae:.4f}")

    # 7) Residual boxplot
    df_res["residual"] = df_res["y_true"] - df_res["y_pred"]
    plt.figure(figsize=(10, 6))
    df_res.boxplot(column="residual", by="pos", rot=45)
    plt.title("Residuals by Position (test set)")
    plt.suptitle("")
    plt.xlabel("Position")
    plt.ylabel("Residual (true - pred)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    plt.savefig(REPORT_PATH)
    print(f"Residual boxplot saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
