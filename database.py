import sqlite3
import os
import hashlib

def get_db_connection():
    conn = sqlite3.connect('deepfake_detection.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with necessary tables"""
    conn = get_db_connection()
    
    # Create users table
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create detections table to store detection history
    conn.execute('''
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_path TEXT NOT NULL,
        result TEXT NOT NULL,
        confidence REAL NOT NULL,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Register a new user"""
    conn = get_db_connection()
    hashed_password = hash_password(password)
    
    try:
        conn.execute(
            'INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
            (username, hashed_password, email)
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        # User already exists
        success = False
    
    conn.close()
    return success

def validate_user(username, password):
    """Validate user credentials"""
    conn = get_db_connection()
    hashed_password = hash_password(password)
    
    user = conn.execute(
        'SELECT * FROM users WHERE username = ? AND password = ?',
        (username, hashed_password)
    ).fetchone()
    
    conn.close()
    return user

def save_detection(user_id, image_path, result, confidence):
    """Save detection results to database"""
    conn = get_db_connection()
    
    conn.execute(
        'INSERT INTO detections (user_id, image_path, result, confidence) VALUES (?, ?, ?, ?)',
        (user_id, image_path, result, confidence)
    )
    
    conn.commit()
    conn.close()

def get_user_detections(user_id):
    """Get detection history for a user"""
    conn = get_db_connection()
    
    detections = conn.execute(
        'SELECT * FROM detections WHERE user_id = ? ORDER BY detected_at DESC',
        (user_id,)
    ).fetchall()
    
    conn.close()
    return detections

# Initialize database when this module is imported
if not os.path.exists('deepfake_detection.db'):
    init_db()
