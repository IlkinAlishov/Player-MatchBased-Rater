#!/usr/bin/env python3
#ingest_data.py


import os
import pandas as pd
import sqlite3

def main():
    RAW_CSV    = os.path.join("data", "raw", "data_football_ratings.csv")
    SQLITE_DB  = os.path.join("data", "raw", "database.sqlite")
    OUTPUT_CSV = os.path.join("data", "processed", "player_ratings_merged.csv")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 1) Load & filter ratings CSV
    ratings_df = pd.read_csv(RAW_CSV)
    ratings_df = ratings_df[ratings_df["rater"] == "WhoScored"].copy()
    ratings_df.drop(columns=["rater", "is_human"], inplace=True)
    ratings_df.rename(columns={
        "player": "player_name",
        "original_rating": "rating"
    }, inplace=True)
    ratings_df["date"] = pd.to_datetime(ratings_df["date"], format="%d/%m/%Y")

    # 2) Load player attributes from SQLite
    conn = sqlite3.connect(SQLITE_DB)
    player_attrs = pd.read_sql_query("""
        SELECT
          p.player_name,
          pa.date            AS attr_date,
          pa.overall_rating,
          pa.potential,
          pa.acceleration,
          pa.sprint_speed,
          pa.finishing,
          pa.short_passing,
          pa.stamina,
          pa.agility,
          pa.vision,
          pa.dribbling,
          pa.marking,
          pa.interceptions,
          pa.heading_accuracy,
          pa.positioning
        FROM Player AS p
        JOIN Player_Attributes AS pa
          ON p.player_api_id = pa.player_api_id
    """, conn)
    conn.close()
    player_attrs["attr_date"] = pd.to_datetime(player_attrs["attr_date"])

    # 3) Sort by the 'on' key only (required for merge_asof)
    ratings_df = ratings_df.sort_values("date").reset_index(drop=True)
    player_attrs = player_attrs.sort_values("attr_date").reset_index(drop=True)

    # 4) Perform the as-of merge, grouping by player_name
    merged_df = pd.merge_asof(
        left=ratings_df,
        right=player_attrs,
        left_on="date",
        right_on="attr_date",
        by="player_name",
        direction="backward"
    )
    merged_df.drop(columns=["attr_date"], inplace=True)

    # 5) (Optional) Drop rows without any matched attributes
    # merged_df = merged_df[merged_df["overall_rating"].notna()]

    # 6) Save the result
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Merged data saved to {OUTPUT_CSV} — shape {merged_df.shape}")

if __name__ == "__main__":
    main()
