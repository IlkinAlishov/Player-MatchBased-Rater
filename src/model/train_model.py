#!/usr/bin/env python3
# train_model.py

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # Paths
    FEAT_CSV    = os.path.join("data", "processed", "player_ratings_features.csv")
    TARGET_CSV  = os.path.join("data", "processed", "player_ratings_target.csv")
    MODEL_PATH  = os.path.join("models", "rating_model.pkl")
    METRICS_TXT = os.path.join("reports", "metrics.txt")
    FI_PNG      = os.path.join("reports", "feature_importances.png")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_TXT), exist_ok=True)

    # 1) Load data
    X = pd.read_csv(FEAT_CSV)
    y = pd.read_csv(TARGET_CSV)["rating"].values

    # 1.1) Clean infinite and NaN values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    X = X.clip(lower=-1e6, upper=1e6).astype(np.float64)

    # 2) Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3) Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model artifact
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 4) Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    # Compute RMSE manually to avoid sklearn version issues
    rmse = mean_squared_error(y_test, y_pred)
    rmse = rmse ** 0.5

    # Write metrics to file
    with open(METRICS_TXT, "w") as f:
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
    print(f"Metrics written to {METRICS_TXT}")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # 5) Feature importances
    importances = model.feature_importances_
    feat_names = X.columns
    top_idx = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(8, 6))
    plt.title("Top 15 Feature Importances")
    plt.barh(
        [feat_names[i] for i in top_idx[::-1]],
        importances[top_idx][::-1]
    )
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FI_PNG)
    plt.close()
    print(f"Feature importances plot saved to {FI_PNG}")

if __name__ == '__main__':
    main()
