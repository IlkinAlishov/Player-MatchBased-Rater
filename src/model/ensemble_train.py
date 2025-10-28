#ensemble_train.py

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from lightgbm import LGBMRegressor

def main():
    # Determine project structure paths
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_file = os.path.join(project_root, 'data', 'raw', 'data_football_ratings.csv')
    
    # Load the dataset
    df = pd.read_csv(data_file)
    target_column = 'target'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Prepare features and target
    X = df.drop(columns=[target_column], errors='ignore')
    if 'pos' in X.columns:
        X = X.drop(columns=['pos'])
    y = df[target_column]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize individual regressors
    rf = RandomForestRegressor(random_state=42)
    lgbm = LGBMRegressor(random_state=42)
    ensemble = VotingRegressor([('rf', rf), ('lgbm', lgbm)])
    
    # Train the ensemble
    ensemble.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = ensemble.predict(X_test)
    
    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Ensemble Model MAE:  {mae:.4f}")
    print(f"Ensemble Model RMSE: {rmse:.4f}")
    
    # Save the trained ensemble model
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(ensemble, os.path.join(models_dir, 'voting_ensemble.pkl'))
    print("Trained ensemble model saved to 'models/voting_ensemble.pkl'.")

if __name__ == "__main__":
    main()
