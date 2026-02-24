import pandas as pd
import numpy as np
import argparse
import joblib
import logging
import os
import shap
import matplotlib.pyplot as plt

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(data_dir):
    """Loads the processed train dataset for SHAP background data."""
    train_path = os.path.join(data_dir, 'train.csv')
    try:
        train_df = pd.read_csv(train_path)
        return train_df.iloc[:, :-1] # Return only features
    except FileNotFoundError as e:
        logging.error(f"Error loading train dataset: {e}")
        return None

def main(data_dir):
    logging.info(f"Loading background data from {data_dir}...")
    X_train = load_data(data_dir)
    
    if X_train is None:
        return

    # Load model
    model_path = 'models/xgb_model.joblib'
    try:
        model = joblib.load(model_path)
        logging.info("Successfully loaded XGBoost model.")
    except FileNotFoundError:
        logging.error(f"Model not found at {model_path}. Please run train.py first.")
        return

    plots_dir = os.path.join('outputs', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    logging.info("Initializing SHAP TreeExplainer...")
    # Explain the model predictions using SHAP
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    sample_size = min(500, len(X_train))
    X_sample = shap.utils.sample(X_train, sample_size)
    shap_values = explainer(X_sample)

    # Feature Importance Bar Chart
    logging.info("Generating SHAP Feature Importance Bar Chart...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Overall Feature Importance for Mobile Price", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'shap_importance_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # SHAP Summary Plot
    logging.info("Generating SHAP Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Impact of Features on Mobile Price Predictions", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # SHAP Dependence Plot
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_feature_idx = np.argsort(mean_abs_shap)[-1]
    top_feature_name = X_train.columns[top_feature_idx]

    logging.info(f"Generating SHAP Dependence Plot for top feature: {top_feature_name}...")
    plt.figure()
    # We use shap.dependence_plot instead of plots.scatter for better older-version compatibility
    shap.dependence_plot(top_feature_name, shap_values.values, X_sample, show=False)
    plt.title(f"How {top_feature_name} Affects Mobile Price", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'shap_dependence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Successfully saved all SHAP explanation plots to {plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SHAP Explanations")
    parser.add_argument('--data', type=str, default='data/processed', help="Directory containing preprocessed train.csv")
    args = parser.parse_args()
    
    main(args.data)
