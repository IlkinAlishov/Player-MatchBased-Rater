#!/usr/bin/env python3

import os, pandas as pd, datetime as dt, textwrap

PROC_DIR = os.path.join('data', 'processed')

# Identify canonical filenames used in the pipeline
FEATURE_FILE = 'player_ratings_features.csv'
TARGET_FILE  = 'player_ratings_target.csv'

if not os.path.isdir(PROC_DIR):
    raise SystemExit(f"Processed directory not found: {PROC_DIR}")

print("Processed CSV summary\n" + "="*40)
for fname in sorted(os.listdir(PROC_DIR)):
    if not fname.endswith('.csv'):  # skip non‑csv
        continue
    fpath = os.path.join(PROC_DIR, fname)
    mtime = dt.datetime.fromtimestamp(os.path.getmtime(fpath)).strftime('%Y‑%m‑%d %H:%M:%S')
    df = pd.read_csv(fpath, nrows=5)  
    full_shape = pd.read_csv(fpath, usecols=[0]).shape  
    tag = ''
    if fname == FEATURE_FILE:
        tag = '<< FEATURES (X) used for model'
    elif fname == TARGET_FILE:
        tag = '<< TARGET (y) used for model'
    print(f"{fname:30}  shape={full_shape[0]}x{len(df.columns)}  modified={mtime} {tag}")
    # show first 5 columns
    cols_preview = ', '.join(df.columns[:5]) + ('...' if len(df.columns) > 5 else '')
    print("   cols: " + textwrap.shorten(cols_preview, width=120))

print("\nNote: The model is always trained on \"player_ratings_features.csv\" (X) and \"player_ratings_target.csv\" (y).\n")
