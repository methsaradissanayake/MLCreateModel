import pandas as pd
import numpy as np
import argparse
import joblib
import logging
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_price(price_str):
    """
    Cleans the currency values (Rs, commas, Lakh, Mn) and converts to numeric.
    """
    if pd.isna(price_str):
        return np.nan
        
    price_str = str(price_str).lower().replace('rs', '').replace(',', '').strip()
    
    # Handle multipliers
    multiplier = 1
    if 'lakh' in price_str or 'laks' in price_str:
        multiplier = 100000
        price_str = price_str.replace('lakh', '').replace('laks', '').strip()
    elif 'mn' in price_str or 'million' in price_str:
        multiplier = 1000000
        price_str = price_str.replace('mn', '').replace('million', '').strip()
        
    try:
        # Extract digits and decimal point
        val = re.findall(r"[-+]?\d*\.\d+|\d+", price_str)
        if val:
            return float(val[0]) * multiplier
        return np.nan
    except:
        return np.nan

def extract_features_from_title(title):
    """
    Extracts Brand, Model, Condition, and Storage from the ad title.
    Example: 'Apple iPhone 14 Pro Max 128GB (Used)'
    """
    title = str(title) if pd.notna(title) else ""
    
    # Extract condition
    condition = 'Unknown'
    if '(Used)' in title:
        condition = 'Used'
        title = title.replace('(Used)', '')
    elif '(Brand New)' in title:
        condition = 'New'
        title = title.replace('(Brand New)', '')
        
    # Extract storage
    storage = np.nan
    storage_match = re.search(r'(\d+)\s*(GB|TB)', title, re.IGNORECASE)
    if storage_match:
        val = int(storage_match.group(1))
        unit = storage_match.group(2).upper()
        storage = val * 1024 if unit == 'TB' else val
        # Remove matched storage from title to help brand/model extraction
        title = title.replace(storage_match.group(0), '')

    # Extract brand and model
    parts = title.strip().split()
    brand = parts[0] if len(parts) > 0 else "Unknown"
    model = " ".join(parts[1:]).strip() if len(parts) > 1 else "Unknown"
    
    # Clean up trailing non-alphanumeric chars
    model = re.sub(r'[^a-zA-Z0-9 ]+$', '', model).strip()
    if not model:
        model = "Unknown"
        
    return pd.Series({'Brand': brand, 'Model': model, 'Condition': condition, 'Storage_GB': storage})

def main(input_path, output_path):
    logging.info(f"Loading data from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_path}")
        return

    # Target column detection
    target_col = 'price'
    if target_col not in df.columns:
        possible_targets = [c for c in df.columns if 'price' in c.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            logging.info(f"Target column 'price' not found. Auto-detected '{target_col}' as target.")
        else:
            logging.error("Could not detect a target price column. Exiting.")
            return

    # Clean price column
    logging.info("Cleaning price column...")
    df['Price_Cleaned'] = df[target_col].apply(clean_price)
    
    # Drop rows without a valid price
    initial_len = len(df)
    df.dropna(subset=['Price_Cleaned'], inplace=True)
    logging.info(f"Dropped {initial_len - len(df)} rows with missing/invalid prices. Remaining: {len(df)}")

    logging.info("Filtering out extreme price outliers (e.g., < Rs 1000 or > Rs 2,000,000)...")
    df = df[(df['Price_Cleaned'] >= 1000) & (df['Price_Cleaned'] <= 2000000)]

    logging.info("Extracting features from title...")
    extracted = df['title'].apply(extract_features_from_title)
    df = pd.concat([df, extracted], axis=1)

    # Fill storage with median storage by Brand if available, else overall median
    df['Storage_GB'] = df.groupby('Brand')['Storage_GB'].transform(lambda x: x.fillna(x.median()) if not x.isna().all() else x)
    df['Storage_GB'].fillna(df['Storage_GB'].median(), inplace=True)

    # Categorical Encoding
    logging.info("Encoding categorical features...")
    cat_columns = ['Brand', 'Model', 'Condition', 'location', 'membershipLevel']
    encoders = {}
    
    # Clean up rare brands/models to avoid high cardinality noise
    brand_counts = df['Brand'].value_counts()
    rare_brands = brand_counts[brand_counts < 10].index
    df.loc[df['Brand'].isin(rare_brands), 'Brand'] = 'Other'
    
    model_counts = df['Model'].value_counts()
    rare_models = model_counts[model_counts < 5].index
    df.loc[df['Model'].isin(rare_models), 'Model'] = 'Other'

    for col in cat_columns:
        if col in df.columns:
            le = LabelEncoder()
            # Handle NaNs
            df[col] = df[col].fillna('Missing').astype(str)
            df[f"{col}_Encoded"] = le.fit_transform(df[col])
            encoders[col] = le

    # Save encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/encoders.joblib')
    logging.info("Saved label encoders to models/encoders.joblib")

    # Select final features
    features = [f"{c}_Encoded" for c in cat_columns if c in df.columns] + ['Storage_GB']
    X = df[features]
    y = df['Price_Cleaned']

    logging.info(f"Final feature set: {features}")

    # Train/Validation/Test Split (70-15-15)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)

    # Save data splits
    output_dir = output_path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Save the splits to separate CSVs for clear downstream usage
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    
    val_df = pd.concat([X_val, y_val], axis=1)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    # Save a reference dataset mapping encoded values back for the UI
    reference_df = df[['Brand', 'Model', 'Condition', 'location', 'Price_Cleaned', 'title']].drop_duplicates(subset=['Brand', 'Model'])
    reference_df.to_csv(os.path.join(output_dir, 'reference.csv'), index=False)

    logging.info(f"Preprocessing complete. Saved splits to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Mobile Price Dataset")
    parser.add_argument('--input', type=str, required=True, help="Path to raw CSV dataset")
    parser.add_argument('--output', type=str, required=True, help="Path to save processed CSVs")
    args = parser.parse_args()
    
    main(args.input, args.output)
