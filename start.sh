#!/bin/bash
set -e

echo "--- Backdoor AI Server Startup Script ---"
echo "Running environment checks..."

# Check Python version
python_version=$(python --version)
echo "Python version: $python_version"

# Check storage paths
echo "Checking storage paths..."
BASE_DIR=${KOYEB_STORAGE_PATH:-/var/koyeb/storage}
echo "Base storage path: $BASE_DIR"

# Ensure directories exist
mkdir -p $BASE_DIR/data
mkdir -p $BASE_DIR/models
mkdir -p $BASE_DIR/training_data
mkdir -p $BASE_DIR/nltk_data

# Check permissions
echo "Checking directory permissions..."
ls -la $BASE_DIR
echo "Data directory:"
ls -la $BASE_DIR/data

# Check if database can be initialized (without running the app)
echo "Checking database initialization..."
python -c "
import os
import sqlite3
from utils.db_helpers import init_db
BASE_DIR = os.environ.get('KOYEB_STORAGE_PATH', '/var/koyeb/storage')
DB_PATH = os.path.join(BASE_DIR, 'data', 'interactions.db')
print(f'Database path: {DB_PATH}')
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
try:
    init_db(DB_PATH)
    print('Database initialized successfully')
    # Check if we can create a test table
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS startup_test (id INTEGER PRIMARY KEY)')
    cursor.execute('DROP TABLE startup_test')
    conn.close()
    print('Database connection test successful')
except Exception as e:
    print(f'Error initializing database: {e}')
    raise
"

# Download NLTK data if needed
echo "Checking NLTK data..."
python -c "
import os
import nltk
NLTK_DATA_PATH = os.path.join(os.environ.get('KOYEB_STORAGE_PATH', '/var/koyeb/storage'), 'nltk_data')
nltk.data.path.append(NLTK_DATA_PATH)
print(f'NLTK data path: {NLTK_DATA_PATH}')
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=False)
        print(f'NLTK {resource} available')
    except Exception as e:
        print(f'Error downloading NLTK {resource}: {e}')
"

# Set Flask debugging mode for better error reports
export FLASK_ENV=development
export FLASK_DEBUG=1

# Start the application with Gunicorn with detailed logging
echo "Starting Gunicorn with detailed logging..."
exec gunicorn --bind 0.0.0.0:${PORT:-8000} \
    --workers 1 \
    --threads 2 \
    --timeout 60 \
    --log-level debug \
    --error-logfile - \
    --access-logfile - \
    --capture-output \
    app:app
