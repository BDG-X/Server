#!/usr/bin/env python
"""
System check utility for Backdoor AI server deployment

This script helps diagnose deployment environments by checking:
- Environment variables
- Directory permissions
- Database access
- System resources
"""

import os
import sys
import platform
import sqlite3
import json
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check environment variables and platform info"""
    env_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "hostname": platform.node(),
        "environment": "production" if os.getenv("KOYEB_APP_NAME") or os.getenv("RENDER_SERVICE_ID") else "development",
        "environment_variables": {
            "PORT": os.getenv("PORT"),
            "KOYEB_APP_NAME": os.getenv("KOYEB_APP_NAME"),
            "KOYEB_SERVICE_NAME": os.getenv("KOYEB_SERVICE_NAME"),
            "KOYEB_STORAGE_PATH": os.getenv("KOYEB_STORAGE_PATH"),
            "RENDER_SERVICE_ID": os.getenv("RENDER_SERVICE_ID"),
            "RENDER_DISK_PATH": os.getenv("RENDER_DISK_PATH")
        }
    }
    return env_info

def check_storage():
    """Check storage paths and permissions"""
    
    # Determine the storage base directory
    if os.getenv("KOYEB_STORAGE_PATH"):
        base_dir = os.getenv("KOYEB_STORAGE_PATH")
        storage_type = "Koyeb"
    elif os.getenv("RENDER_DISK_PATH"):
        base_dir = os.getenv("RENDER_DISK_PATH")
        storage_type = "Render"
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        storage_type = "Local"
    
    # Define required directories
    required_dirs = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "training_data"),
        os.path.join(base_dir, "nltk_data")
    ]
    
    dir_info = []
    for directory in required_dirs:
        dir_exists = os.path.exists(directory)
        # Create directory if it doesn't exist
        if not dir_exists:
            try:
                os.makedirs(directory, exist_ok=True)
                dir_exists = True
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                
        # Check permissions
        try:
            readable = os.access(directory, os.R_OK) if dir_exists else False
            writable = os.access(directory, os.W_OK) if dir_exists else False
            
            # Try writing a test file
            write_test = False
            if dir_exists and writable:
                test_file = os.path.join(directory, ".test_write")
                try:
                    with open(test_file, 'w') as f:
                        f.write("test")
                    write_test = True
                    os.remove(test_file)
                except Exception as e:
                    logger.error(f"Write test failed: {e}")
        except Exception as e:
            readable = False
            writable = False
            write_test = False
            logger.error(f"Permission check failed: {e}")
        
        dir_info.append({
            "path": directory,
            "exists": dir_exists,
            "readable": readable,
            "writable": writable,
            "write_test": write_test
        })
    
    # Check disk space
    try:
        disk_usage = shutil.disk_usage(base_dir)
        disk_info = {
            "total": disk_usage.total,
            "used": disk_usage.used,
            "free": disk_usage.free,
            "percent_used": round(disk_usage.used / disk_usage.total * 100, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get disk usage: {e}")
        disk_info = {"error": str(e)}
    
    return {
        "storage_type": storage_type,
        "base_directory": base_dir,
        "directories": dir_info,
        "disk_usage": disk_info
    }

def check_database():
    """Check database connectivity and structure"""
    
    # Determine the database path
    if os.getenv("KOYEB_STORAGE_PATH"):
        base_dir = os.getenv("KOYEB_STORAGE_PATH")
    elif os.getenv("RENDER_DISK_PATH"):
        base_dir = os.getenv("RENDER_DISK_PATH")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    db_path = os.path.join(base_dir, "data", "interactions.db")
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Check if the database exists and can be accessed
    db_exists = os.path.exists(db_path)
    
    conn = None
    tables = []
    row_counts = {}
    initialization_result = None
    
    try:
        # Try to connect to database
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Count rows in main tables
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_counts[table] = cursor.fetchone()[0]
            except:
                row_counts[table] = -1  # Error counting rows
        
        # Initialize database if not initialized
        if not tables:
            try:
                from utils.db_helpers import init_db
                init_db(db_path)
                initialization_result = "Database initialized successfully"
                
                # Check tables again
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                initialization_result = f"Database initialization failed: {str(e)}"
        
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return {
            "database_path": db_path,
            "exists": db_exists,
            "accessible": False,
            "error": str(e)
        }
    finally:
        if conn:
            conn.close()
    
    return {
        "database_path": db_path,
        "exists": db_exists,
        "accessible": True,
        "tables": tables,
        "row_counts": row_counts,
        "initialization_result": initialization_result
    }

def main():
    """Run all system checks and print results"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "environment": check_environment(),
        "storage": check_storage(),
        "database": check_database()
    }
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Determine overall status
    storage_ok = all(d.get("writable", False) for d in results["storage"]["directories"])
    database_ok = results["database"].get("accessible", False)
    
    if storage_ok and database_ok:
        print("\n✅ System check passed. The server should function correctly.")
        return 0
    else:
        print("\n❌ System check failed.")
        if not storage_ok:
            print("   - Storage issues detected. Check permissions.")
        if not database_ok:
            print("   - Database issues detected. Check connectivity.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
