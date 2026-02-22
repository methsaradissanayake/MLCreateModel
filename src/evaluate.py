import pandas as pd
import numpy as np
import argparse
import joblib
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_dir):
    """Loads the processed test dataset."""
    test_path = os.path.join(data_dir, 'test.csv')
    try:
        test_df = pd.read_csv(test_path)
        return test_df
    except FileNotFoundError as e:
        logging.error(f"Error loading test dataset: {e}")
        return None

def generate_plots(y_true, y_pred, output_dir):
    """Generates and saves evaluation plots with human-readable labels."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")

    # 1. Predicted vs Actual Plot
    logging.info("Generating Predicted vs Actual plot...")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='royalblue')
    
    # Add perfect prediction line
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
    
    plt.title('Predicted Mobile Phone Price vs Actual Mobile Phone Price', fontsize=14, pad=15)
    plt.xlabel('Actual Price (LKR)', fontsize=12)
    plt.ylabel('Predicted Price (LKR)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predicted_vs_actual.png'), dpi=300)
    plt.close()

    # 2. Residual Histogram
    logging.info("Generating Residuals Histogram...")
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple', bins=50)
    
    plt.title('Distribution of Price Prediction Errors', fontsize=14, pad=15)
    plt.xlabel('Prediction Error (Actual Price - Predicted Price)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residual_histogram.png'), dpi=300)
    plt.close()


def main(data_dir):
    logging.info(f"Loading test data from {data_dir}...")
    test_df = load_data(data_dir)
    
    if test_df is None:
        return

    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    # Load model
    model_path = 'models/xgb_model.joblib'
    try:
        model = joblib.load(model_path)
        logging.info("Successfully loaded XGBoost model.")
    except FileNotFoundError:
        logging.error(f"Model not found at {model_path}. Please run train.py first.")
        return

    # Predict
    logging.info("Making predictions on the test set...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Evaluation Results - RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")

    # Save metrics
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2_Score': float(r2)
    }

    # Save as JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save as CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics_table.csv'), index=False)
    logging.info(f"Saved metrics to {output_dir}")

    # Generate Plots
    generate_plots(y_test, y_pred, output_dir)
    logging.info("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate XGBoost Model")
    parser.add_argument('--data', type=str, default='data/processed', help="Directory containing preprocessed test.csv")
    args = parser.parse_args()
    
    main(args.data)
