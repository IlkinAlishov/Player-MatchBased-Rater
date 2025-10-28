#!/usr/bin/env python3
# prepare_data.py

import os
import pandas as pd
import numpy as np # Added for inf handling

def main():
    # ── Paths ───────────────────────────────────────────────────────────────────
    # Input file from ingest_data.py
    INPUT_CSV  = os.path.join("data", "processed", "player_ratings_merged.csv")
    # Outputs for model training
    FEAT_CSV   = os.path.join("data", "processed", "player_ratings_features.csv")
    TARGET_CSV = os.path.join("data", "processed", "player_ratings_target.csv")
    os.makedirs(os.path.dirname(FEAT_CSV), exist_ok=True)

    # ── 1) Load merged data ────────────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded merged data shape: {df.shape}")

    # ── 2) CLEANING LOGIC (From old clean_data.py) ──────────────────────────────
    # Drop or fill missing values

    # Corrected code for prepare_data.py (Use direct assignment)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    for col in numeric_cols:
        if col != 'rating':  # don't fill target
            # Calculate the median and assign the filled result back to the column
            df[col] = df[col].fillna(df[col].median()) # <-- FIX: Removed inplace=True

    # Drop any rows still containing nulls
    df.dropna(inplace=True)

    # Ensure correct data types (for columns needed in feature engineering)
    df['competition'] = df['competition'].astype('category')
    df['pos'] = df['pos'].astype('category')
    df['pos_role'] = df['pos_role'].astype('category')
    df['win'] = df['win'].astype(int)
    df['lost'] = df['lost'].astype(int)
    df['is_home_team'] = df['is_home_team'].astype(int)

    # ── 3) FEATURE ENGINEERING LOGIC (Per-90 and Encoding) ──────────────────────
    stats_to_scale = [
        'goals', 'assists', 'shots_ontarget', 'shots_offtarget', 'shotsblocked',
        'chances2score', 'drib_success', 'drib_unsuccess', 'keypasses',
        'touches', 'passes_acc', 'passes_inacc', 'crosses_acc', 'crosses_inacc',
        'lballs_acc', 'lballs_inacc', 'grduels_w', 'grduels_l',
        'aerials_w', 'aerials_l', 'poss_lost', 'fouls', 'wasfouled',
        'clearances', 'stop_shots', 'interceptions', 'tackles', 'dribbled_past',
        'tballs_acc', 'tballs_inacc', 'countattack', 'offsides'
    ]
    stats_present = [col for col in stats_to_scale if col in df.columns]

    # Compute per-90 metrics (handle division by zero if 'minutesPlayed' is 0)
    #df['minutesPlayed'].replace(0, np.nan, inplace=True) # Replace 0 with NaN for division
    df['minutesPlayed'] = df['minutesPlayed'].replace(0, np.nan) # FIX: Removed inplace=True
    for col in stats_present:
        # Scale stats only where minutesPlayed is valid
        df[f"{col}_per90"] = df[col] / df["minutesPlayed"] * 90

    # Drop the original count columns
    df.drop(columns=stats_present, inplace=True)
    df.drop(columns=["minutesPlayed"], inplace=True) # Drop 'minutesPlayed' now that per-90 is done

    # Encode categorical features
    cat_cols = [c for c in ["competition", "pos", "pos_role"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)


    # ... (Continuing from Section 3, where encoding is complete)

    # ── 4) Separate target and features & Final Cleanup ────────────────────────

    if "rating" not in df.columns:
        raise KeyError("Expected 'rating' column in data as the target.")

    # 1. Extract the target 'y' from the main DataFrame 'df' FIRST
    y = df["rating"].values

    # 2. Create the feature DataFrame 'X' by dropping the target and meta-columns from 'df'
    # Drop: target ('rating'), original time/identifier columns ('date', 'match'), 
    # and the problematic text columns ('player_name', 'team') that caused the TypeError.
    X = df.drop(
        columns=["rating", "date", "match", "player_name", "team"], 
        errors="ignore"
    ) 

    # 3. Final Sanity Check: Ensure 'X' contains only numeric types
    # (The previous steps should have handled this, but this is good practice 
    # if you want to be absolutely sure no object columns remain, uncomment the line below)
    # X = X.select_dtypes(include=np.number)

    # ── 5) Save out features and target ─────────────────────────────────────────
    X.to_csv(FEAT_CSV, index=False)
    pd.DataFrame({"rating": y}).to_csv(TARGET_CSV, index=False)
    print(f"Features saved to {FEAT_CSV} (shape: {X.shape})")
    print(f"Target saved to {TARGET_CSV} (shape: {y.shape})")


if __name__ == "__main__":
    main()

