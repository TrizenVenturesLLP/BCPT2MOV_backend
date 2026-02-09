"""
Migration script to add transport_cost column to products table
Run this once: python migrate_add_transport_cost.py
"""
import sqlite3
import os

DB_PATH = 'agri_insights.db'

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} does not exist. It will be created automatically on next backend start.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if column already exists
        cursor.execute("PRAGMA table_info(products)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'transport_cost' in columns:
            print("Column 'transport_cost' already exists. Migration not needed.")
            conn.close()
            return
        
        # Add the column
        print("Adding transport_cost column to products table...")
        cursor.execute("ALTER TABLE products ADD COLUMN transport_cost REAL")
        conn.commit()
        print("✓ Migration successful! transport_cost column added.")
        
    except Exception as e:
        print(f"✗ Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
