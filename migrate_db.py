import sqlite3

def migrate_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE predictions ADD COLUMN image_path TEXT")
        print("Successfully added image_path column to predictions table.")
    except sqlite3.OperationalError as e:
        print(f"Error: {e} (Column may already exist)")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    migrate_db()