import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import requests

# Set paths
DATA_DB = 'joined_data.db'
MODELS_DIR = 'models'

# Ensure the models folder exists
os.makedirs(MODELS_DIR, exist_ok=True)

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

def train_model(X_train, y_train, model_type):
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=800)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=800)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=800)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.fit(X_train, y_train)
    return model

def save_model(model, table_name, model_type):
    filename = f"{MODELS_DIR}/{table_name}_{model_type}.joblib"
    joblib.dump(model, filename)
    print(f"Saved model: {filename}")

def main():
    download_database()
    tables = get_table_names(DATA_DB)

    for table in tables:
        print(f"Processing table: {table}")
        df = load_data_from_table(DATA_DB, table).dropna()

        if 'date' in df.index.names:
            df.reset_index(inplace=True)

        X = df.drop(columns=['Date', 'target_n7d'])
        y = df['target_n7d']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=True)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model = train_model(X_train_scaled, y_train, model_type)
            save_model(model, table, model_type)

if __name__ == "__main__":
    main()
