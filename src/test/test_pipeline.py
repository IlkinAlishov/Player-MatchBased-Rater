#test_pipeline.py

import os, joblib, pandas as pd, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")

def test_processed_files_exist():
    for fname in ["player_ratings_features.csv",
                  "player_ratings_target.csv",
                  "player_ratings_cleaned.csv"]:
        path = os.path.join(PROC, fname)
        assert os.path.isfile(path) and os.path.getsize(path) > 0, f"{fname} missing"

def test_model_predicts():
    model = joblib.load(os.path.join(ROOT, "models", "rating_model.pkl"))
    X = pd.read_csv(os.path.join(PROC, "player_ratings_features.csv")).head(3)
    preds = model.predict(X)
    assert preds.shape == (3,)
    assert np.all((preds >= 1) & (preds <= 10)), "predictions out of 1-10 range"

if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__]))
