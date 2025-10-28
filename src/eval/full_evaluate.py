#!/usr/bin/env python3
#full_evaluate.py
import os, argparse, numpy as np, pandas as pd, joblib, textwrap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC  = os.path.join(ROOT, "data", "processed")
FEAT  = os.path.join(PROC, "player_ratings_features.csv")
TARGET= os.path.join(PROC, "player_ratings_target.csv")
MERGED = os.path.join(PROC, "player_ratings_merged.csv")
REPORT= os.path.join(ROOT, "reports", "full_eval.txt")
HIST  = os.path.join(ROOT, "reports", "residual_hist.png")

BANDS = [
    ("≈6", 5.75, 6.25),
    ("≈7", 6.75, 7.25),
    ("≥8", 7.75, 10.1),
]

def header(title):
    return f"\n{title}\n" + "-" * len(title)


def main(model_path):
    X = pd.read_csv(FEAT)
    y = pd.read_csv(TARGET)["rating"].values
    # Get metadata from the original merged file BEFORE it was cleaned/transformed
    meta = pd.read_csv(MERGED)[["pos", "competition", "rating"]]
# def main(model_path):
#     X = pd.read_csv(FEAT)
#     y = pd.read_csv(TARGET)["rating"].values
#     meta = pd.read_csv(CLEAN)[["pos", "competition", "rating"]]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)

    X_tr, X_te, y_tr, y_te, meta_tr, meta_te = train_test_split(
        X, y, meta, test_size=0.2, random_state=42
    )
    # align indices
    meta_te = meta_te.reset_index(drop=True)
    X_te    = X_te.reset_index(drop=True)

    model = joblib.load(model_path)
    preds = model.predict(X_te)

    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))

    lines = [
        f"MODEL : {os.path.basename(model_path)}",
        f"SAMPLES: train={len(X_tr)}, test={len(X_te)}",
        f"MAE    : {mae:.4f}",
        f"RMSE   : {rmse:.4f}",
    ]

    # by rating band
    lines.append(header("MAE by rating band"))
    for label, lo, hi in BANDS:
        mask = (meta_te["rating"] >= lo) & (meta_te["rating"] < hi)
        if mask.any():
            lines.append(f"{label:>3}: {mean_absolute_error(y_te[mask], preds[mask]):.4f}  (n={mask.sum()})")

    # by position
    lines.append(header("MAE by position"))
    for p in sorted(meta_te["pos"].unique()):
        mask = meta_te["pos"] == p
        lines.append(f"{p:>3}: {mean_absolute_error(y_te[mask], preds[mask]):.4f}  (n={mask.sum()})")

    # by competition (top 10 rows)
    lines.append(header("MAE by competition (top 10)"))
    comp_mae = []
    for comp in meta_te["competition"].unique():
        mask = meta_te["competition"] == comp
        comp_mae.append((comp, mean_absolute_error(y_te[mask], preds[mask]), mask.sum()))
    for comp, m, n in sorted(comp_mae, key=lambda x: x[1])[:10]:
        lines.append(f"{comp[:25]:25}: {m:.4f}  (n={n})")

    # residual histogram
    resid = y_te - preds
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=30, edgecolor='k')
    plt.title('Residuals (true - pred)')
    plt.xlabel('Residual'); plt.ylabel('Count'); plt.tight_layout()
    os.makedirs(os.path.dirname(HIST), exist_ok=True)
    plt.savefig(HIST); plt.close()

    # write report
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f"\nReport -> {REPORT}\nHistogram -> {HIST}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to .pkl model')
    main(parser.parse_args().model)
