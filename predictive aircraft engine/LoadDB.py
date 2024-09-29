import pandas as pd
import sqlite3

class CSVToSQLite:
    def __init__(self, db_name='data.db'):
        self.db_name = db_name

    def upload_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            print("Preview of the CSV file:")
            print(df.head())
            table_name = input("Enter the table name to save the data: ")
            self.save_to_db(df, table_name)
        except Exception as e:
            print(f"An error occurred: {e}")

    def save_to_db(self, df, table_name):
        try:
            conn = sqlite3.connect(self.db_name)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            print(f"Data saved to table '{table_name}' in the database '{self.db_name}'")
        except Exception as e:
            print(f"An error occurred while saving to the database: {e}")

if __name__ == "__main__":
    csv_to_sqlite = CSVToSQLite()
    file_path = "new_train_data.csv"
    csv_to_sqlite.upload_csv(file_path)
