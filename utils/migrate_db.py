import os
import sqlite3
from db_utils import DB_PATH

def migrate_database():
    """
    Migrates the database schema to add the email field to existing students table
    if it doesn't already exist.
    """
    print(f"Checking for database at {DB_PATH}")
    
    if not os.path.exists(DB_PATH):
        print("Database not found. No migration needed.")
        return
    
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if email column exists in students table
    cursor.execute("PRAGMA table_info(students)")
    columns = cursor.fetchall()
    column_names = [column[1] for column in columns]
    
    if "email" not in column_names:
        print("Adding 'email' column to students table...")
        try:
            cursor.execute("ALTER TABLE students ADD COLUMN email TEXT")
            conn.commit()
            print("Migration successful! Email column added.")
        except sqlite3.Error as e:
            print(f"Error during migration: {e}")
    else:
        print("Email column already exists. No migration needed.")
    
    conn.close()

if __name__ == "__main__":
    migrate_database() 