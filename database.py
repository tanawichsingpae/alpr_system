import sqlite3

DB_NAME = "parking.db"

def init_db():

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS parking_records (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        plate TEXT,
        province TEXT,

        entry_time TEXT,
        exit_time TEXT,

        duration_minutes INTEGER,
        fee INTEGER,

        entry_image TEXT,
        exit_image TEXT,

        plate_image TEXT
    )
    """)

    conn.commit()
    conn.close()