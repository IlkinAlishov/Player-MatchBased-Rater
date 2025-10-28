#!/usr/bin/env python3
# predict_rating.py


import os, argparse, pandas as pd, numpy as np, joblib

def load_features(path):
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    X_new = load_features(args.input)
    model = joblib.load(args.model)

    preds = model.predict(X_new.values)
    out_df = X_new.copy()
    out_df["predicted_rating"] = preds
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(preds)} predictions → {args.output}")

if __name__ == "__main__":
    main()
