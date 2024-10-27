import os
import sqlite3
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

DATA_DB = 'joined_data.db'
MODELS_DIR = 'models'
PREDICTIONS_DB = 'data_v1/predictions.db'

os.makedirs('data', exist_ok=True)

def load_data_from_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def load_model(table_name, model_type):
    filename = f"{MODELS_DIR}/{table_name}_{model_type}.joblib"
    return joblib.load(filename)

def save_predictions_to_db(table_name, predictions_df):
    conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(f'predictions_{table_name}', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved predictions for {table_name} to {PREDICTIONS_DB}")

def main():
    conn = sqlite3.connect(DATA_DB)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
    conn.close()

    for table in tables:
        print(f"Processing predictions for table: {table}")
        df = load_data_from_table(DATA_DB, table).dropna()

        X = df.drop(columns=['Date', 'target_n7d'])
        y = df['target_n7d']
        dates = df['Date']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        predictions_df = pd.DataFrame({'date': dates, 'actual': y})

        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model = load_model(table, model_type)
            predictions = model.predict(X_scaled)
            predictions_df[f'predicted_{model_type}'] = predictions

        save_predictions_to_db(table, predictions_df)

if __name__ == "__main__":
    main()
