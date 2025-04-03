import sqlite3
import os
import logging
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interactions (
            id TEXT PRIMARY KEY,
            device_id TEXT,
            timestamp TEXT,
            user_message TEXT,
            ai_response TEXT,
            detected_intent TEXT,
            confidence_score REAL,
            app_version TEXT,
            model_version TEXT,
            os_version TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            interaction_id TEXT PRIMARY KEY,
            rating INTEGER,
            comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES interactions (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            version TEXT PRIMARY KEY,
            path TEXT,
            accuracy REAL,
            training_data_size INTEGER,
            training_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # New table for uploaded models
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_models (
            id TEXT PRIMARY KEY,
            device_id TEXT,
            app_version TEXT,
            description TEXT,
            file_path TEXT,
            file_size INTEGER,
            original_filename TEXT,
            upload_date TEXT,
            incorporated_in_version TEXT,
            incorporation_status TEXT DEFAULT 'pending', -- pending, processing, incorporated, failed
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Table for tracking ensemble models
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ensemble_models (
            ensemble_version TEXT PRIMARY KEY,
            description TEXT,
            component_models TEXT, -- JSON array of model IDs that make up this ensemble
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Table for tracking training jobs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_jobs (
            id TEXT PRIMARY KEY,
            device_id TEXT,
            status TEXT DEFAULT 'queued', -- queued, processing, completed, failed
            start_time TEXT,
            estimated_completion_time TEXT,
            actual_completion_time TEXT,
            progress REAL DEFAULT 0.0,
            result_model_version TEXT,
            error_message TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT, -- 'uploaded_model', 'json_data', 'combined'
            source_id TEXT, -- ID of the uploaded model or JSON data file
            metadata TEXT -- JSON string with additional metadata
        )
    ''')
    # Table for storing JSON training data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data (
            id TEXT PRIMARY KEY,
            device_id TEXT,
            app_version TEXT,
            data_path TEXT, -- Path to the stored JSON file
            file_size INTEGER,
            description TEXT,
            upload_date TEXT,
            processed_status TEXT DEFAULT 'pending', -- pending, processed, failed
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")

def store_interactions(db_path, data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        conn.execute("BEGIN TRANSACTION")
        for interaction in data['interactions']:
            cursor.execute('''
                INSERT OR REPLACE INTO interactions 
                (id, device_id, timestamp, user_message, ai_response, detected_intent, confidence_score, app_version, model_version, os_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction['id'],
                data['deviceId'],
                interaction['timestamp'],
                interaction['userMessage'],
                interaction['aiResponse'],
                interaction['detectedIntent'],
                interaction['confidenceScore'],
                data['appVersion'],
                data['modelVersion'],
                data['osVersion']
            ))
            if 'feedback' in interaction and interaction['feedback']:
                cursor.execute('''
                    INSERT OR REPLACE INTO feedback 
                    (interaction_id, rating, comment)
                    VALUES (?, ?, ?)
                ''', (
                    interaction['id'],
                    interaction['feedback']['rating'],
                    interaction['feedback'].get('comment')
                ))
        conn.commit()
        logger.info(f"Stored {len(data['interactions'])} interactions from device {data['deviceId']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing interactions: {str(e)}")
        raise
    finally:
        conn.close()

def store_uploaded_model(db_path, device_id, app_version, description, file_path, file_size, original_filename):
    """
    Store metadata about an uploaded model in the database
    """
    model_id = str(uuid.uuid4())
    upload_date = datetime.now().isoformat()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO uploaded_models
            (id, device_id, app_version, description, file_path, file_size, original_filename, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            device_id,
            app_version,
            description,
            file_path,
            file_size,
            original_filename,
            upload_date
        ))
        conn.commit()
        logger.info(f"Stored metadata for uploaded model: {model_id} from device {device_id}")
        return model_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing uploaded model metadata: {str(e)}")
        raise
    finally:
        conn.close()

def store_training_data(db_path, device_id, app_version, description, data_path, file_size):
    """
    Store metadata about uploaded JSON training data
    """
    data_id = str(uuid.uuid4())
    upload_date = datetime.now().isoformat()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO training_data
            (id, device_id, app_version, description, data_path, file_size, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_id,
            device_id,
            app_version,
            description,
            data_path,
            file_size,
            upload_date
        ))
        conn.commit()
        logger.info(f"Stored metadata for training data: {data_id} from device {device_id}")
        return data_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing training data metadata: {str(e)}")
        raise
    finally:
        conn.close()

def create_training_job(db_path, device_id, source_type, source_id, metadata=None):
    """
    Create a new training job record in the database
    """
    job_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    # Estimate completion time based on data size and complexity
    # This is a simplified estimate, could be improved based on historical data
    estimated_minutes = 5  # Default 5 minutes
    
    if metadata and 'data_size' in metadata:
        # Adjust estimate based on data size
        try:
            data_size = int(metadata['data_size'])
            if data_size > 1000000:  # >1MB
                estimated_minutes = 15
            elif data_size > 5000000:  # >5MB
                estimated_minutes = 30
        except (ValueError, TypeError):
            pass
    
    estimated_completion = (datetime.now() + timedelta(minutes=estimated_minutes)).isoformat()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO training_jobs
            (id, device_id, status, created_at, estimated_completion_time, source_type, source_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id,
            device_id,
            'queued',
            created_at,
            estimated_completion,
            source_type,
            source_id,
            json.dumps(metadata) if metadata else None
        ))
        conn.commit()
        logger.info(f"Created training job {job_id} for device {device_id}")
        return job_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating training job: {str(e)}")
        raise
    finally:
        conn.close()

def update_training_job_status(db_path, job_id, status, progress=None, model_version=None, error_message=None):
    """
    Update the status of a training job
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        params = []
        sql_parts = []
        
        if status:
            sql_parts.append("status = ?")
            params.append(status)
            
            # Set timestamps based on status
            if status == 'processing' and not cursor.execute("SELECT start_time FROM training_jobs WHERE id = ?", (job_id,)).fetchone()[0]:
                sql_parts.append("start_time = ?")
                params.append(datetime.now().isoformat())
            elif status in ('completed', 'failed'):
                sql_parts.append("actual_completion_time = ?")
                params.append(datetime.now().isoformat())
        
        if progress is not None:
            sql_parts.append("progress = ?")
            params.append(progress)
            
            # Update estimated completion time based on progress
            if 0 < progress < 1.0:
                cursor.execute("SELECT start_time FROM training_jobs WHERE id = ?", (job_id,))
                start_time_str = cursor.fetchone()[0]
                if start_time_str:
                    start_time = datetime.fromisoformat(start_time_str)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if progress > 0.05:  # Only estimate after some meaningful progress
                        total_estimated = elapsed / progress
                        remaining = total_estimated - elapsed
                        new_estimate = (datetime.now() + timedelta(seconds=remaining)).isoformat()
                        sql_parts.append("estimated_completion_time = ?")
                        params.append(new_estimate)
        
        if model_version:
            sql_parts.append("result_model_version = ?")
            params.append(model_version)
            
        if error_message:
            sql_parts.append("error_message = ?")
            params.append(error_message)
            
        if not sql_parts:
            return
            
        sql = f"UPDATE training_jobs SET {', '.join(sql_parts)} WHERE id = ?"
        params.append(job_id)
        
        cursor.execute(sql, tuple(params))
        conn.commit()
        logger.info(f"Updated training job {job_id} status to {status if status else 'unchanged'}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating training job status: {str(e)}")
        raise
    finally:
        conn.close()

def get_training_job(db_path, job_id):
    """
    Get details of a training job
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM training_jobs
            WHERE id = ?
        ''', (job_id,))
        job = cursor.fetchone()
        return dict(job) if job else None
    except Exception as e:
        logger.error(f"Error retrieving training job {job_id}: {str(e)}")
        return None
    finally:
        conn.close()

def get_device_training_jobs(db_path, device_id, limit=10):
    """
    Get training jobs for a specific device
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM training_jobs
            WHERE device_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (device_id, limit))
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error retrieving training jobs for device {device_id}: {str(e)}")
        return []
    finally:
        conn.close()

def update_model_incorporation_status(db_path, model_id, status, version=None):
    """
    Update the status of an uploaded model's incorporation into the ensemble
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        if version:
            cursor.execute('''
                UPDATE uploaded_models
                SET incorporation_status = ?, incorporated_in_version = ?
                WHERE id = ?
            ''', (status, version, model_id))
        else:
            cursor.execute('''
                UPDATE uploaded_models
                SET incorporation_status = ?
                WHERE id = ?
            ''', (status, model_id))
        conn.commit()
        logger.info(f"Updated incorporation status for model {model_id} to {status}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating model incorporation status: {str(e)}")
        raise
    finally:
        conn.close()

def get_pending_uploaded_models(db_path):
    """
    Get all uploaded models that haven't been incorporated into an ensemble yet
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM uploaded_models
            WHERE incorporation_status IN ('pending', 'processing')
        ''')
        return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error retrieving pending uploaded models: {str(e)}")
        return []
    finally:
        conn.close()

# Add missing import at the top
import json
