import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import requests

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

def extract_percentiles(model, X_scaled):
    """Extract predictions and calculate 5th and 95th percentiles for models with estimators."""
    all_preds = []

    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            if hasattr(est, 'predict') and callable(getattr(est, 'predict', None)):
                try:
                    pred = est.predict(X_scaled)
                    if isinstance(pred, (np.ndarray, list)):
                        all_preds.append(pred)
                    else:
                        print(f"Skipping estimator with non-numeric predictions: {type(est)}")
                except Exception as e:
                    print(f"Error during prediction with estimator {type(est)}: {e}")
            else:
                print(f"Skipping an estimator without 'predict' or incorrect structure: {type(est)}")

    if not all_preds:
        raise ValueError("No valid estimators with 'predict' method found.")

    all_preds_df = pd.DataFrame(all_preds).T
    p5 = all_preds_df.quantile(0.05, axis=1)
    p95 = all_preds_df.quantile(0.95, axis=1)
    return p5, p95

def save_predictions_to_db(predictions_df, table_name):
    """Save predictions to the predictions database."""
    conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(f'predictions_{table_name}', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Predictions saved to database for table: {table_name}")

def predict_random_forest(X_scaled, predictions_df, model_path):
    """Predict with RandomForest model, add results to predictions_df."""
    model = joblib.load(model_path)
    predictions_df['Predicted_random_forest'] = model.predict(X_scaled)
    print("RandomForest predictions added.")

def predict_gradient_boosting(X_scaled, predictions_df, model_path):
    """Predict with GradientBoosting model, add results and percentiles to predictions_df."""
    model = joblib.load(model_path)
    predictions_df['Predicted_gradient_boosting'] = model.predict(X_scaled)
    try:
        p5, p95 = extract_percentiles(model, X_scaled)
        predictions_df['5th_Percentile_gradient_boosting'] = p5
        predictions_df['95th_Percentile_gradient_boosting'] = p95
        print("Successfully calculated 5th and 95th percentiles for GradientBoosting.")
    except ValueError as e:
        print(f"Warning: {e} for GradientBoosting model")

def predict_xgboost(X_scaled, predictions_df, model_path):
    """Predict with XGBoost model, add results and percentiles to predictions_df."""
    model = joblib.load(model_path)
    predictions_df['Predicted_xgboost'] = model.predict(X_scaled)
    try:
        p5, p95 = extract_percentiles(model, X_scaled)
        predictions_df['5th_Percentile_xgboost'] = p5
        predictions_df['95th_Percentile_xgboost'] = p95
        print("Successfully calculated 5th and 95th percentiles for XGBoost.")
    except ValueError as e:
        print(f"Warning: {e} for XGBoost model")

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
        X_scaled = scaler.fit_transform(X)

        model_paths = {
            'random_forest': os.path.join(MODELS_DIR, f"{table}_random_forest.joblib"),
            'gradient_boosting': os.path.join(MODELS_DIR, f"{table}_gradient_boosting.joblib"),
            'xgboost': os.path.join(MODELS_DIR, f"{table}_xgboost.joblib"),
        }

        if os.path.exists(model_paths['random_forest']):
            predict_random_forest(X_scaled, predictions_df, model_paths['random_forest'])
        else:
            print(f"RandomForest model not found for {table}")

        if os.path.exists(model_paths['gradient_boosting']):
            predict_gradient_boosting(X_scaled, predictions_df, model_paths['gradient_boosting'])
        else:
            print(f"GradientBoosting model not found for {table}")

        if os.path.exists(model_paths['xgboost']):
            predict_xgboost(X_scaled, predictions_df, model_paths['xgboost'])
        else:
            print(f"XGBoost model not found for {table}")

        save_predictions_to_db(predictions_df, table)

if __name__ == "__main__":
    main()
