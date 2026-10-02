import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "toxic_app.db"


# IF NOT EXISTS so restarting the app doesn't crash or lose the old rows
def create_table():
    connection = sqlite3.connect(DB_PATH)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_name TEXT,
            input_type TEXT,
            user_text TEXT,
            image_caption TEXT,
            label TEXT
        )
    """)
    connection.commit()
    connection.close()


def save_record(user_name, input_type, user_text, image_caption, label):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ? placeholders so user input is never run as sql
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        "INSERT INTO records (timestamp, user_name, input_type, user_text, image_caption, label) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, user_name, input_type, user_text, image_caption, label),
    )
    connection.commit()
    connection.close()


# newest first
def get_all_records():
    connection = sqlite3.connect(DB_PATH)
    records = pd.read_sql_query("SELECT * FROM records ORDER BY id DESC", connection)
    connection.close()
    return records


# test with two fake rows, delete toxic_app.db afterwards
if __name__ == "__main__":
    create_table()
    save_record("Nagham", "text", "How to kill someone?", None, "Violent Crimes")
    save_record("Nagham", "image", None, "two cats sleeping on a couch", "Non-Violent Crimes")
    print(get_all_records().to_string())
