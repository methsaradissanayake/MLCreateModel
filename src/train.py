import pandas as pd
import numpy as np
import argparse
import joblib
import logging
import os
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_dir):
    """Loads the processed train and validation datasets."""
    train_path = os.path.join(data_dir, 'train.csv')
    val_path = os.path.join(data_dir, 'val.csv')
    
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        return train_df, val_df
    except FileNotFoundError as e:
        logging.error(f"Error loading datasets: {e}")
        return None, None

def main(data_dir):
    logging.info(f"Loading preprocessed data from {data_dir}...")
    train_df, val_df = load_data(data_dir)
    
    if train_df is None or val_df is None:
        return

    # Assume the last column is the target (Price_Cleaned)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    
    X_val = val_df.iloc[:, :-1]
    y_val = val_df.iloc[:, -1]

    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Validation data shape: {X_val.shape}")

    # 1. Initialize XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    # 2. Hyperparameter Grid for RandomizedSearchCV
    # Justification: RandomizedSearchCV is faster than GridSearchCV for finding near-optimal 
    # hyperparams across a large search space, which saves time while maintaining good performance.
    param_dist = {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    logging.info("Starting RandomizedSearchCV for hyperparameter tuning...")
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=20, # Number of random combinations to try
        scoring='neg_root_mean_squared_error',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Fit the search on the training data
    search.fit(X_train, y_train)
    
    best_params = search.best_params_
    logging.info(f"Best hyperparameters found: {best_params}")

    # 3. Train final model with early stopping on validation set
    logging.info("Training final model with early stopping...")
    final_model = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50 # Stop if validation error doesn't improve for 50 rounds
    )

    # Fit with evaluation set
    # Using eval_set to track performance and trigger early stopping
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50
    )

    # 4. Save the trained model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/xgb_model.joblib'
    joblib.dump(final_model, model_path)
    logging.info(f"Successfully saved trained model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Model")
    parser.add_argument('--data', type=str, required=True, help="Directory containing preprocessed train.csv and val.csv")
    args = parser.parse_args()
    
    main(args.data)
