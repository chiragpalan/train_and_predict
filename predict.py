import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import numpy as np

# Paths and configurations
DATA_DB = 'joined_data.db'
PREDICTIONS_DB = 'data/predictions.db'
MODELS_DIR = 'models_v1'
DATA_FOLDER = 'data_v1'

# Ensure folders exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

def download_database():
    """Download the database from the GitHub repository."""
    url = 'https://raw.githubusercontent.com/chiragpalan/final_project/main/database/joined_data.db'
    response = requests.get(url)
    if response.status_code == 200:
        with open(DATA_DB, 'wb') as f:
            f.write(response.content)
        print("Database downloaded successfully.")
    else:
        raise Exception("Failed to download the database.")

def get_table_names(db_path):
    """Retrieve all table names from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def load_data_from_table(db_path, table_name):
    """Load data from a specific table."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def validate_and_clean_data(X):
    """Remove infinite and NaN values and replace them with zeroes."""
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def extract_predictions_from_estimators(model, X_scaled):
    """Extract predictions and calculate percentiles, handling different model structures."""
    all_preds = []

    # Check if model has estimators (like RandomForest or GradientBoosting)
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            # Ensure the estimator has a predict method and is not an ndarray
            if hasattr(est, 'predict') and not isinstance(est, np.ndarray):
                all_preds.append(est.predict(X_scaled))
            else:
                print(f"Skipping an estimator without 'predict': {type(est)}")

    if not all_preds:
        raise ValueError("No valid estimators with 'predict' method found.")

    # Convert predictions to a DataFrame and calculate percentiles
    all_preds_df = pd.DataFrame(all_preds).T  # Transpose to match input shape
    p5 = all_preds_df.quantile(0.05, axis=1)
    p95 = all_preds_df.quantile(0.95, axis=1)
    return p5, p95

def save_predictions_to_db(predictions_df, table_name):
    """Save predictions to the predictions database."""
    conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(f'predictions_{table_name}', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Predictions saved to database for table: {table_name}")

def main():
    download_database()
    tables = get_table_names(DATA_DB)

    for table in tables:
        print(f"Processing table: {table}")
        df = load_data_from_table(DATA_DB, table).dropna()

        if 'Date' not in df.columns:
            raise KeyError("The 'Date' column is missing from the data.")

        X = df.drop(columns=['Date', 'target_n7d'], errors='ignore')
        y_actual = df['target_n7d']
        dates = df['Date']

        predictions_df = pd.DataFrame({'Date': dates, 'Actual': y_actual})

        scaler = StandardScaler()
        X = validate_and_clean_data(X)
        X_scaled = scaler.fit_transform(X)

        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model_path = os.path.join(MODELS_DIR, f"{table}_{model_type}.joblib")
            print(f"Loading model from: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = joblib.load(model_path)
            predictions = model.predict(X_scaled)
            predictions_df[f'Predicted_{model_type}'] = predictions

            # Handle models with or without individual estimators
            try:
                if hasattr(model, 'estimators_'):
                    p5, p95 = extract_predictions_from_estimators(model, X_scaled)
                    predictions_df[f'5th_Percentile_{model_type}'] = p5
                    predictions_df[f'95th_Percentile_{model_type}'] = p95
            except ValueError as e:
                print(f"Warning: {e} for model {model_type}")

        save_predictions_to_db(predictions_df, table)

if __name__ == "__main__":
    main()
