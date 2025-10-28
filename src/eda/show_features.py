#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np



def main():
    # Path to cleaned data
    DATA_CSV = os.path.join('data', 'processed', 'player_ratings_cleaned.csv')

    # Load the cleaned dataset
    df = pd.read_csv(DATA_CSV)

    # 1) Basic info
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", df.columns.tolist())
    print()

    # 2) Missing value report
    missing_pct = df.isna().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if len(missing_pct) > 0:
        print("Columns with missing values (%):")
        print(missing_pct.to_string())
    else:
        print("No missing values in the cleaned dataset.")
    print()

    # 3) Correlation with target
    if 'rating' not in df.columns:
        raise KeyError("The target column 'rating' was not found in the dataset.")

    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    # Compute Pearson correlation with target
    corr_with_target = numeric_df.corr()['rating'].drop('rating').abs()
    corr_sorted = corr_with_target.sort_values(ascending=False)

    print("Top 10 features by absolute Pearson correlation with rating:")
    top10 = corr_sorted.head(10)
    for feat, corr_val in top10.items():
        print(f"  {feat}: {corr_val:.3f}")
    print()

    # 4) Descriptive stats for top features
    print("Descriptive statistics for top features:")
    stats = df[top10.index].describe()
    print(stats.to_string())

if __name__ == '__main__':
    main()
