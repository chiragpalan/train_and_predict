import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import requests

# Paths and configurations
DATA_DB = 'joined_data.db'
PREDICTIONS_DB = 'data/predictions.db'
MODELS_DIR = 'models_v1'
DATA_FOLDER = 'data_v1'

# Ensure folders exist, including the data folder for the predictions database
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(PREDICTIONS_DB), exist_ok=True)  # Ensure data folder exists

# Confirm directory structure
print(f"Models directory exists: {os.path.isdir(MODELS_DIR)}")
print(f"Data folder exists: {os.path.isdir(DATA_FOLDER)}")
print(f"Predictions database path: {PREDICTIONS_DB}")

def download_database():
    url = 'https://raw.githubusercontent.com/chiragpalan/final_project/main/database/joined_data.db'
    response = requests.get(url)
    if response.status_code == 200:
        with open(DATA_DB, 'wb') as f:
            f.write(response.content)
        print("Database downloaded successfully.")
    else:
        raise Exception("Failed to download the database.")

def get_table_names(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def load_data_from_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def clean_data(X):
    """Check for infinity, NaN, or very large values in X and clean them."""
    if not np.isfinite(X.values).all():
        print("Warning: X contains NaN, infinity, or very large values. Cleaning data...")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def extract_percentiles(predictions):
    """Calculate 5th and 95th percentiles from an array of predictions."""
    p5 = np.percentile(predictions, 5, axis=0)
    p95 = np.percentile(predictions, 95, axis=0)
    return p5, p95

def gradient_boosting_predictions(model, X_scaled):
    n_estimators = model.n_estimators
    individual_predictions = np.zeros((X_scaled.shape[0], n_estimators))
    
    # Collect predictions from each estimator
    for i in range(n_estimators):
        individual_predictions[:, i] = model.estimators_[i, 0].predict(X_scaled)

    # Calculate percentiles
    p5 = np.percentile(individual_predictions, 5, axis=1)
    p95 = np.percentile(individual_predictions, 95, axis=1)
    main_prediction = model.predict(X_scaled)
    
    return main_prediction, p5, p95

def save_predictions_to_db(predictions_df, prediction_table_name):
    try:
        # Open a connection to the predictions database
        conn = sqlite3.connect(PREDICTIONS_DB)
        print(f"Saving predictions to table: {prediction_table_name} in {PREDICTIONS_DB}")
        
        # Write the DataFrame to a new table
        predictions_df.to_sql(prediction_table_name, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Predictions successfully saved to table: {prediction_table_name}")
    except Exception as e:
        print(f"Error while saving to database: {e}")

def process_table(table):
    df = load_data_from_table(DATA_DB, table).dropna()
    if 'Date' not in df.columns:
        raise KeyError("The 'Date' column is missing from the data.")

    X = df.drop(columns=['Date', 'target_n7d'], errors='ignore')
    y_actual = df['target_n7d']
    dates = df['Date']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = clean_data(X_scaled)  # Clean the scaled data

    model_types = ['random_forest', 'gradient_boosting', 'xgboost']
    prediction_functions = {
        'random_forest': random_forest_predictions,
        'gradient_boosting': gradient_boosting_predictions,
        'xgboost': xgboost_predictions
    }

    for model_type in model_types:
        model_path = os.path.join(MODELS_DIR, f"{table}_{model_type}.joblib")
        print(f"Loading model from: {model_path}")

        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            continue

        model = joblib.load(model_path)

        # Predict using the appropriate function
        prediction_func = prediction_functions[model_type]
        try:
            main_prediction, p5, p95 = prediction_func(model, X_scaled)
            
            # Prepare DataFrame to save predictions
            predictions_df = pd.DataFrame({
                'Table_Name': table,
                'Date': dates,
                'Actual': y_actual,
                f'Predicted_{model_type}': main_prediction,
                f'5th_Percentile_{model_type}': p5,
                f'95th_Percentile_{model_type}': p95
            })

            # Define table name for saving predictions
            prediction_table_name = f"prediction_{table}_{model_type}"
            save_predictions_to_db(predictions_df, prediction_table_name)
            
        except Exception as e:
            print(f"Error predicting with {model_type}: {e}")

def main():
    download_database()
    tables = get_table_names(DATA_DB)

    for table in tables:
        print(f"Processing table: {table}")
        try:
            process_table(table)
        except Exception as e:
            print(f"Error processing table {table}: {e}")

    # Check if predictions database file exists after processing
    if os.path.exists(PREDICTIONS_DB):
        print(f"Predictions database created at: {PREDICTIONS_DB}")
    else:
        print("Predictions database was not created.")

if __name__ == "__main__":
    main()
