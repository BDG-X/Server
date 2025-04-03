import sqlite3
import os
import logging
import uuid
import json
from datetime import datetime, timedelta
import re
import hashlib
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_LENGTH = 10000  # Maximum allowed length for text fields
MAX_DB_RETRIES = 3       # Maximum number of retries for database operations
TIMEOUT_SECONDS = 30     # Timeout for database operations
VALID_STATUS_VALUES = {'pending', 'processing', 'incorporated', 'failed', 'queued', 'completed'}

def get_db_connection(db_path, timeout=TIMEOUT_SECONDS, detect_types=sqlite3.PARSE_DECLTYPES):
    """
    Create a database connection with proper settings and error handling
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        # Connect with enhanced settings
        conn = sqlite3.connect(
            db_path, 
            timeout=timeout,
            detect_types=detect_types, 
            isolation_level=None  # Autocommit mode for better control
        )
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        # Safe text encoding
        conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def execute_with_retry(func, max_retries=MAX_DB_RETRIES):
    """
    Execute a database function with retry logic for transient errors
    """
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            # Only retry on database locked errors
            if "database is locked" in str(e) and attempt < max_retries - 1:
                wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Database locked, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            # Don't retry on other errors
            logger.error(f"Database error: {str(e)}")
            raise

def sanitize_string(s, max_length=MAX_TEXT_LENGTH):
    """Sanitize string input to prevent SQL injection and ensure valid UTF-8"""
    if s is None:
        return None
    
    # Convert to string if not already
    if not isinstance(s, str):
        s = str(s)
    
    # Truncate to max length
    s = s[:max_length]
    
    # Remove any potentially dangerous characters
    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
    
    return s

def validate_uuid(id_str):
    """Validate that a string is a properly formatted UUID"""
    if not id_str:
        return False
    
    try:
        uuid_obj = uuid.UUID(id_str)
        return str(uuid_obj) == id_str
    except (ValueError, AttributeError, TypeError):
        return False
    
def hash_file(file_path, algorithm='sha256'):
    """Generate a hash for a file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
            
    return hash_obj.hexdigest()

def init_db(db_path):
    """Initialize the database with all required tables"""
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Begin transaction for atomic database creation
        conn.execute("BEGIN")
        
        # Create interactions table with indices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
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
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_device_id ON interactions(device_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_detected_intent ON interactions(detected_intent)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at)')
        
        # Create feedback table with indices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                interaction_id TEXT PRIMARY KEY,
                rating INTEGER,
                comment TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id) ON DELETE CASCADE
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)')
        
        # Create model versions table with indices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                version TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                accuracy REAL,
                file_hash TEXT,
                training_data_size INTEGER,
                training_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_versions_training_date ON model_versions(training_date)')
        
        # Table for uploaded models with indices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_models (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                app_version TEXT,
                description TEXT,
                file_path TEXT NOT NULL,
                file_hash TEXT,
                file_size INTEGER,
                original_filename TEXT,
                upload_date TEXT NOT NULL,
                incorporated_in_version TEXT,
                incorporation_status TEXT DEFAULT 'pending' CHECK(incorporation_status IN ('pending', 'processing', 'incorporated', 'failed')),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (incorporated_in_version) REFERENCES model_versions (version) ON DELETE SET NULL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uploaded_models_device_id ON uploaded_models(device_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uploaded_models_status ON uploaded_models(incorporation_status)')
        
        # Table for tracking ensemble models
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ensemble_models (
                ensemble_version TEXT PRIMARY KEY,
                description TEXT,
                component_models TEXT, -- JSON array of model IDs that make up this ensemble
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ensemble_version) REFERENCES model_versions (version) ON DELETE CASCADE
            )
        ''')
        
        # Table for tracking training jobs with indices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_jobs (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                status TEXT DEFAULT 'queued' CHECK(status IN ('queued', 'processing', 'completed', 'failed')),
                start_time TEXT,
                estimated_completion_time TEXT,
                actual_completion_time TEXT,
                progress REAL DEFAULT 0.0 CHECK(progress >= 0 AND progress <= 1.0),
                result_model_version TEXT,
                error_message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source_type TEXT CHECK(source_type IN ('uploaded_model', 'json_data', 'combined', 'scheduled')),
                source_id TEXT,
                metadata TEXT, -- JSON string with additional metadata
                FOREIGN KEY (result_model_version) REFERENCES model_versions (version) ON DELETE SET NULL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_jobs_device_id ON training_jobs(device_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON training_jobs(created_at)')
        
        # Table for storing JSON training data with indices
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL,
                app_version TEXT,
                data_path TEXT NOT NULL, -- Path to the stored JSON file
                file_hash TEXT,
                file_size INTEGER,
                description TEXT,
                upload_date TEXT NOT NULL,
                processed_status TEXT DEFAULT 'pending' CHECK(processed_status IN ('pending', 'processed', 'failed')),
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_data_device_id ON training_data(device_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_data_status ON training_data(processed_status)')
        
        conn.execute("COMMIT")
        logger.info(f"Database initialized successfully at {db_path}")
        
    except sqlite3.Error as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Database initialization error: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def store_interactions(db_path, data):
    """
    Store interaction data in the database with validation and sanitization
    """
    if not data or not isinstance(data, dict) or 'interactions' not in data:
        raise ValueError("Invalid data format: missing interactions")
    
    if not data.get('deviceId'):
        raise ValueError("Invalid data format: missing deviceId")
    
    # Validate and sanitize device ID
    device_id = sanitize_string(data.get('deviceId'), 100)
    app_version = sanitize_string(data.get('appVersion'), 50)
    model_version = sanitize_string(data.get('modelVersion'), 50)
    os_version = sanitize_string(data.get('osVersion'), 50)
    
    interactions = data.get('interactions', [])
    if not interactions or not isinstance(interactions, list):
        raise ValueError("Invalid data format: interactions must be a non-empty list")
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        
        # Begin transaction
        conn.execute("BEGIN IMMEDIATE")
        
        # Process interactions
        successful_count = 0
        for interaction in interactions:
            try:
                # Validate required fields
                if not interaction.get('id'):
                    logger.warning("Skipping interaction with missing ID")
                    continue
                
                # Sanitize inputs
                interaction_id = sanitize_string(interaction.get('id'), 100)
                timestamp = sanitize_string(interaction.get('timestamp'), 50)
                user_message = sanitize_string(interaction.get('userMessage'), MAX_TEXT_LENGTH)
                ai_response = sanitize_string(interaction.get('aiResponse'), MAX_TEXT_LENGTH)
                detected_intent = sanitize_string(interaction.get('detectedIntent'), 100)
                
                # Validate numeric fields
                try:
                    confidence_score = float(interaction.get('confidenceScore', 0))
                    if not (0 <= confidence_score <= 1):
                        confidence_score = 0
                except (ValueError, TypeError):
                    confidence_score = 0
                
                # Insert or replace interaction
                conn.execute('''
                    INSERT OR REPLACE INTO interactions 
                    (id, device_id, timestamp, user_message, ai_response, detected_intent, confidence_score, app_version, model_version, os_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    interaction_id,
                    device_id,
                    timestamp,
                    user_message,
                    ai_response,
                    detected_intent,
                    confidence_score,
                    app_version,
                    model_version,
                    os_version
                ))
                
                # Process feedback if present
                if 'feedback' in interaction and interaction['feedback']:
                    feedback = interaction['feedback']
                    
                    # Validate rating
                    try:
                        rating = int(feedback.get('rating', 0))
                        if not (1 <= rating <= 5):
                            rating = None
                    except (ValueError, TypeError):
                        rating = None
                    
                    comment = sanitize_string(feedback.get('comment'), 1000)
                    
                    # Only insert valid feedback
                    if rating is not None:
                        conn.execute('''
                            INSERT OR REPLACE INTO feedback 
                            (interaction_id, rating, comment)
                            VALUES (?, ?, ?)
                        ''', (
                            interaction_id,
                            rating,
                            comment
                        ))
                
                successful_count += 1
            
            except Exception as e:
                logger.error(f"Error processing interaction {interaction.get('id', 'unknown')}: {str(e)}")
                # Continue with next interaction rather than failing the entire batch
                continue
        
        # Commit transaction
        conn.execute("COMMIT")
        logger.info(f"Successfully stored {successful_count} of {len(interactions)} interactions from device {device_id}")
        
        return successful_count
    
    except sqlite3.Error as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Database error while storing interactions: {str(e)}")
        raise
    except Exception as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Error storing interactions: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def safe_file_operation(func, *args, **kwargs):
    """Execute a file operation with proper error handling and recovery"""
    try:
        return func(*args, **kwargs)
    except (IOError, OSError) as e:
        logger.error(f"File operation error: {str(e)}")
        raise

def store_uploaded_model(db_path, device_id, app_version, description, file_path, file_size, original_filename):
    """
    Store metadata about an uploaded model in the database with validation
    """
    # Validate inputs
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    
    if not original_filename.endswith('.mlmodel'):
        raise ValueError("Invalid file format: must be a .mlmodel file")
    
    # Sanitize inputs
    device_id = sanitize_string(device_id, 100)
    app_version = sanitize_string(app_version, 50)
    description = sanitize_string(description, 1000)
    original_filename = sanitize_string(original_filename, 255)
    
    # Generate unique model ID
    model_id = str(uuid.uuid4())
    upload_date = datetime.now().isoformat()
    
    # Compute file hash for integrity verification
    file_hash = safe_file_operation(hash_file, file_path)
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        
        # Check if the same file has already been uploaded (based on hash)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM uploaded_models WHERE file_hash = ?", (file_hash,))
        existing = cursor.fetchone()
        
        if existing:
            logger.info(f"Duplicate model file detected (hash: {file_hash})")
            # Still proceed with the upload but log it
        
        # Begin transaction
        conn.execute("BEGIN IMMEDIATE")
        
        # Insert model metadata
        conn.execute('''
            INSERT INTO uploaded_models
            (id, device_id, app_version, description, file_path, file_hash, file_size, original_filename, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            device_id,
            app_version,
            description,
            file_path,
            file_hash,
            file_size,
            original_filename,
            upload_date
        ))
        
        # Commit transaction
        conn.execute("COMMIT")
        logger.info(f"Stored metadata for uploaded model: {model_id} from device {device_id}")
        
        return model_id
    
    except sqlite3.Error as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Database error while storing model metadata: {str(e)}")
        raise
    except Exception as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Error storing uploaded model metadata: {str(e)}")
        
        # Clean up file if database operation failed
        try:
            if os.path.exists(file_path):
                logger.info(f"Cleaning up uploaded file after database error: {file_path}")
                os.remove(file_path)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up file: {str(cleanup_error)}")
        
        raise
    finally:
        if conn:
            conn.close()

def store_training_data(db_path, device_id, app_version, description, data_path, file_size):
    """
    Store metadata about uploaded JSON training data with validation
    """
    # Validate inputs
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    
    if not data_path.endswith('.json'):
        raise ValueError("Invalid file format: must be a .json file")
    
    # Validate file is valid JSON
    try:
        with open(data_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON file: file contains invalid JSON format")
    
    # Sanitize inputs
    device_id = sanitize_string(device_id, 100)
    app_version = sanitize_string(app_version, 50)
    description = sanitize_string(description, 1000)
    
    # Generate unique data ID
    data_id = str(uuid.uuid4())
    upload_date = datetime.now().isoformat()
    
    # Compute file hash for integrity verification
    file_hash = safe_file_operation(hash_file, data_path)
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        
        # Check if the same file has already been uploaded (based on hash)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM training_data WHERE file_hash = ?", (file_hash,))
        existing = cursor.fetchone()
        
        if existing:
            logger.info(f"Duplicate training data file detected (hash: {file_hash})")
            # Still proceed with the upload but log it
        
        # Begin transaction
        conn.execute("BEGIN IMMEDIATE")
        
        # Insert training data metadata
        conn.execute('''
            INSERT INTO training_data
            (id, device_id, app_version, description, data_path, file_hash, file_size, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data_id,
            device_id,
            app_version,
            description,
            data_path,
            file_hash,
            file_size,
            upload_date
        ))
        
        # Commit transaction
        conn.execute("COMMIT")
        logger.info(f"Stored metadata for training data: {data_id} from device {device_id}")
        
        return data_id
    
    except sqlite3.Error as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Database error while storing training data metadata: {str(e)}")
        raise
    except Exception as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Error storing training data metadata: {str(e)}")
        
        # Clean up file if database operation failed
        try:
            if os.path.exists(data_path):
                logger.info(f"Cleaning up uploaded file after database error: {data_path}")
                os.remove(data_path)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up file: {str(cleanup_error)}")
        
        raise
    finally:
        if conn:
            conn.close()

def create_training_job(db_path, device_id, source_type, source_id, metadata=None):
    """
    Create a new training job record in the database with validation
    """
    # Validate inputs
    device_id = sanitize_string(device_id, 100)
    
    # Validate source type
    valid_source_types = {'uploaded_model', 'json_data', 'combined', 'scheduled'}
    if source_type not in valid_source_types:
        raise ValueError(f"Invalid source type: {source_type}. Must be one of {valid_source_types}")
    
    # Validate source ID based on source type
    if source_type != 'scheduled':
        source_id = sanitize_string(source_id, 100)
        if not source_id:
            raise ValueError(f"Source ID is required for source type: {source_type}")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    # Process metadata
    if metadata and not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")
    
    # Calculate estimated completion time
    estimated_minutes = 5  # Default 5 minutes
    
    if metadata:
        # If explicit estimate provided, use it
        if 'estimated_minutes' in metadata:
            try:
                estimated_minutes = int(metadata['estimated_minutes'])
                # Ensure reasonable bounds
                if estimated_minutes < 1:
                    estimated_minutes = 1
                elif estimated_minutes > 120:  # Cap at 2 hours
                    estimated_minutes = 120
            except (ValueError, TypeError):
                pass
        
        # Otherwise estimate based on data size
        elif 'data_size' in metadata:
            try:
                data_size = int(metadata['data_size'])
                if data_size > 5000000:  # >5MB
                    estimated_minutes = 30
                elif data_size > 1000000:  # >1MB
                    estimated_minutes = 15
                elif data_size > 100000:  # >100KB
                    estimated_minutes = 10
            except (ValueError, TypeError):
                pass
    
    estimated_completion = (datetime.now() + timedelta(minutes=estimated_minutes)).isoformat()
    metadata_json = json.dumps(metadata) if metadata else None
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        
        # Check for existing jobs for the same source to avoid duplicates
        if source_type != 'scheduled':
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, status FROM training_jobs 
                WHERE source_type = ? AND source_id = ? AND status IN ('queued', 'processing')
            """, (source_type, source_id))
            existing_job = cursor.fetchone()
            
            if existing_job:
                logger.warning(f"Found existing active job {existing_job[0]} for {source_type}:{source_id}")
                # Return existing job ID if it's still in progress
                return existing_job[0]
        
        # Begin transaction
        conn.execute("BEGIN IMMEDIATE")
        
        # Insert training job
        conn.execute('''
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
            metadata_json
        ))
        
        # Commit transaction
        conn.execute("COMMIT")
        logger.info(f"Created training job {job_id} for device {device_id} ({source_type}:{source_id})")
        
        return job_id
    
    except sqlite3.Error as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Database error while creating training job: {str(e)}")
        raise
    except Exception as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Error creating training job: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def update_training_job_status(db_path, job_id, status, progress=None, model_version=None, error_message=None):
    """
    Update the status of a training job with validation
    """
    # Validate job ID
    if not job_id or not validate_uuid(job_id):
        raise ValueError(f"Invalid job ID: {job_id}")
    
    # Validate status
    if status and status not in {'queued', 'processing', 'completed', 'failed'}:
        raise ValueError(f"Invalid status: {status}")
    
    # Validate progress
    if progress is not None:
        try:
            progress = float(progress)
            if not (0 <= progress <= 1):
                raise ValueError(f"Progress must be between 0 and 1, got {progress}")
        except (ValueError, TypeError):
            raise ValueError(f"Invalid progress value: {progress}")
    
    # Sanitize inputs
    if model_version:
        model_version = sanitize_string(model_version, 50)
    
    if error_message:
        error_message = sanitize_string(error_message, 1000)
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        
        # Check if job exists
        cursor = conn.cursor()
        cursor.execute("SELECT id, status FROM training_jobs WHERE id = ?", (job_id,))
        existing_job = cursor.fetchone()
        
        if not existing_job:
            raise ValueError(f"Job not found: {job_id}")
        
        current_status = existing_job[1]
        
        # Validate status transition
        if status and status != current_status:
            # Prevent invalid transitions (e.g., completed -> processing)
            if current_status in {'completed', 'failed'} and status not in {'completed', 'failed'}:
                logger.warning(f"Invalid status transition for job {job_id}: {current_status} -> {status}")
                return
        
        # Begin transaction
        conn.execute("BEGIN IMMEDIATE")
        
        # Build update query
        params = []
        sql_parts = []
        
        if status:
            sql_parts.append("status = ?")
            params.append(status)
            
            # Set timestamps based on status
            if status == 'processing' and current_status != 'processing':
                sql_parts.append("start_time = ?")
                params.append(datetime.now().isoformat())
            elif status in {'completed', 'failed'} and current_status not in {'completed', 'failed'}:
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
                    try:
                        start_time = datetime.fromisoformat(start_time_str)
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if progress > 0.05:  # Only estimate after some meaningful progress
                            total_estimated = elapsed / progress
                            remaining = total_estimated - elapsed
                            # Add a 10% buffer for safety
                            remaining *= 1.1
                            new_estimate = (datetime.now() + timedelta(seconds=remaining)).isoformat()
                            sql_parts.append("estimated_completion_time = ?")
                            params.append(new_estimate)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error calculating estimated completion time: {str(e)}")
        
        if model_version:
            # Verify model version exists
            cursor.execute("SELECT version FROM model_versions WHERE version = ?", (model_version,))
            if cursor.fetchone():
                sql_parts.append("result_model_version = ?")
                params.append(model_version)
            else:
                logger.warning(f"Model version not found: {model_version}")
        
        if error_message:
            sql_parts.append("error_message = ?")
            params.append(error_message)
        
        # Execute update if we have changes
        if sql_parts:
            sql = f"UPDATE training_jobs SET {', '.join(sql_parts)} WHERE id = ?"
            params.append(job_id)
            
            conn.execute(sql, tuple(params))
            
            # Commit transaction
            conn.execute("COMMIT")
            
            status_text = f"status={status}, " if status else ""
            progress_text = f"progress={progress:.2f}, " if progress is not None else ""
            logger.info(f"Updated training job {job_id}: {status_text}{progress_text}model={model_version}")
        else:
            # No changes to make
            conn.execute("ROLLBACK")
    
    except sqlite3.Error as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Database error while updating training job: {str(e)}")
        raise
    except Exception as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Error updating training job status: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def get_training_job(db_path, job_id):
    """
    Get details of a training job with validation
    """
    # Validate job ID
    if not job_id or not validate_uuid(job_id):
        raise ValueError(f"Invalid job ID: {job_id}")
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT j.*, m.path as model_path
            FROM training_jobs j
            LEFT JOIN model_versions m ON j.result_model_version = m.version
            WHERE j.id = ?
        ''', (job_id,))
        
        job = cursor.fetchone()
        
        if not job:
            return None
        
        # Convert Row to dict and process metadata
        job_dict = dict(job)
        
        # Parse metadata JSON if present
        if job_dict.get('metadata'):
            try:
                job_dict['metadata'] = json.loads(job_dict['metadata'])
            except json.JSONDecodeError:
                job_dict['metadata'] = {}
        
        return job_dict
    
    except sqlite3.Error as e:
        logger.error(f"Database error while retrieving training job: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving training job {job_id}: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def get_device_training_jobs(db_path, device_id, limit=10):
    """
    Get training jobs for a specific device with validation
    """
    # Validate inputs
    device_id = sanitize_string(device_id, 100)
    
    try:
        limit = int(limit)
        if limit < 1:
            limit = 10
        elif limit > 100:  # Impose reasonable upper bound
            limit = 100
    except (ValueError, TypeError):
        limit = 10
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute('''
            SELECT j.*, m.path as model_path
            FROM training_jobs j
            LEFT JOIN model_versions m ON j.result_model_version = m.version
            WHERE j.device_id = ?
            ORDER BY j.created_at DESC
            LIMIT ?
        ''', (device_id, limit))
        
        jobs = cursor.fetchall()
        
        # Convert Rows to dicts and process metadata
        result = []
        for job in jobs:
            job_dict = dict(job)
            
            # Parse metadata JSON if present
            if job_dict.get('metadata'):
                try:
                    job_dict['metadata'] = json.loads(job_dict['metadata'])
                except json.JSONDecodeError:
                    job_dict['metadata'] = {}
            
            result.append(job_dict)
        
        return result
    
    except sqlite3.Error as e:
        logger.error(f"Database error while retrieving device training jobs: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving training jobs for device {device_id}: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

def update_model_incorporation_status(db_path, model_id, status, version=None):
    """
    Update the status of an uploaded model's incorporation with validation
    """
    # Validate model ID
    if not model_id or not validate_uuid(model_id):
        raise ValueError(f"Invalid model ID: {model_id}")
    
    # Validate status
    if status not in VALID_STATUS_VALUES:
        raise ValueError(f"Invalid status: {status}")
    
    # Sanitize version if provided
    if version:
        version = sanitize_string(version, 50)
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        
        # Check if model exists
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM uploaded_models WHERE id = ?", (model_id,))
        if not cursor.fetchone():
            raise ValueError(f"Model not found: {model_id}")
        
        # Verify version exists if provided
        if version:
            cursor.execute("SELECT version FROM model_versions WHERE version = ?", (version,))
            if not cursor.fetchone():
                logger.warning(f"Model version not found: {version}, proceeding without linking")
                version = None
        
        # Begin transaction
        conn.execute("BEGIN IMMEDIATE")
        
        # Update model status
        if version:
            conn.execute('''
                UPDATE uploaded_models
                SET incorporation_status = ?, incorporated_in_version = ?
                WHERE id = ?
            ''', (status, version, model_id))
        else:
            conn.execute('''
                UPDATE uploaded_models
                SET incorporation_status = ?
                WHERE id = ?
            ''', (status, model_id))
        
        # Commit transaction
        conn.execute("COMMIT")
        
        version_text = f" (incorporated in {version})" if version else ""
        logger.info(f"Updated incorporation status for model {model_id} to {status}{version_text}")
    
    except sqlite3.Error as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Database error while updating model status: {str(e)}")
        raise
    except Exception as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Error updating model incorporation status: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def get_pending_uploaded_models(db_path):
    """
    Get all uploaded models that haven't been incorporated into an ensemble yet
    """
    conn = None
    try:
        conn = get_db_connection(db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        
        # Get models with hashes and verify file existence
        cursor.execute('''
            SELECT *
            FROM uploaded_models
            WHERE incorporation_status IN ('pending', 'processing')
        ''')
        
        models = cursor.fetchall()
        result = []
        
        for model in models:
            model_dict = dict(model)
            file_path = model_dict.get('file_path')
            
            # Skip models with missing files
            if not os.path.exists(file_path):
                logger.warning(f"Model file not found for model {model_dict['id']}: {file_path}")
                # Update status to failed
                conn.execute(
                    "UPDATE uploaded_models SET incorporation_status = 'failed', error_message = ? WHERE id = ?",
                    ("File not found", model_dict['id'])
                )
                continue
            
            # Verify file hash if available
            if model_dict.get('file_hash'):
                try:
                    current_hash = hash_file(file_path)
                    if current_hash != model_dict['file_hash']:
                        logger.warning(f"File hash mismatch for model {model_dict['id']}")
                        # Mark as failed due to hash mismatch
                        conn.execute(
                            "UPDATE uploaded_models SET incorporation_status = 'failed', error_message = ? WHERE id = ?",
                            ("File hash mismatch", model_dict['id'])
                        )
                        continue
                except Exception as e:
                    logger.error(f"Error verifying file hash: {str(e)}")
            
            result.append(model_dict)
        
        return result
    
    except sqlite3.Error as e:
        logger.error(f"Database error while retrieving pending models: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error retrieving pending uploaded models: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

def get_model_info(db_path, version=None):
    """
    Get detailed information about a specific model version or the latest model
    """
    conn = None
    try:
        conn = get_db_connection(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if version:
            # Get specific version
            cursor.execute("""
                SELECT m.*, e.description as ensemble_description, e.component_models
                FROM model_versions m
                LEFT JOIN ensemble_models e ON m.version = e.ensemble_version
                WHERE m.version = ?
            """, (version,))
        else:
            # Get latest version
            cursor.execute("""
                SELECT m.*, e.description as ensemble_description, e.component_models
                FROM model_versions m
                LEFT JOIN ensemble_models e ON m.version = e.ensemble_version
                ORDER BY m.training_date DESC
                LIMIT 1
            """)
        
        model = cursor.fetchone()
        
        if not model:
            # If no models found, return a default
            if version:
                return None
            else:
                return {
                    'version': '1.0.0',
                    'path': None,
                    'training_date': datetime.now().isoformat()
                }
        
        model_dict = dict(model)
        
        # Parse component models if this is an ensemble
        if model_dict.get('component_models'):
            try:
                model_dict['component_models'] = json.loads(model_dict['component_models'])
            except json.JSONDecodeError:
                model_dict['component_models'] = []
        
        # Verify file exists and is readable
        if model_dict.get('path') and not os.path.exists(model_dict['path']):
            logger.warning(f"Model file not found: {model_dict['path']}")
            model_dict['file_missing'] = True
        
        return model_dict
    
    except sqlite3.Error as e:
        logger.error(f"Database error while retrieving model info: {str(e)}")
        # Return a fallback if database error
        return {
            'version': '1.0.0',
            'path': None,
            'training_date': datetime.now().isoformat(),
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"Error retrieving model info: {str(e)}")
        # Return a fallback on any error
        return {
            'version': '1.0.0',
            'path': None,
            'training_date': datetime.now().isoformat(),
            'error': str(e)
        }
    finally:
        if conn:
            conn.close()

def clean_old_training_data(db_path, days_old=30, dry_run=False):
    """
    Clean up old training data files and database records
    """
    cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
    
    conn = None
    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Find old training data
        cursor.execute("""
            SELECT id, data_path FROM training_data 
            WHERE upload_date < ? AND processed_status != 'pending'
        """, (cutoff_date,))
        
        old_data = cursor.fetchall()
        logger.info(f"Found {len(old_data)} old training data files to clean up")
        
        if dry_run:
            logger.info("Dry run - no files will be deleted")
            return len(old_data)
        
        # Begin transaction
        conn.execute("BEGIN IMMEDIATE")
        
        deleted_count = 0
        for data_id, data_path in old_data:
            try:
                # Delete file if it exists
                if os.path.exists(data_path):
                    os.remove(data_path)
                    logger.info(f"Deleted old training data file: {data_path}")
                
                # Delete database record
                conn.execute("DELETE FROM training_data WHERE id = ?", (data_id,))
                deleted_count += 1
                
            except Exception as e:
                logger.error(f"Error deleting training data {data_id}: {str(e)}")
                continue
        
        # Commit transaction
        conn.execute("COMMIT")
        logger.info(f"Successfully cleaned up {deleted_count} old training data files")
        
        return deleted_count
    
    except sqlite3.Error as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Database error during cleanup: {str(e)}")
        return 0
    except Exception as e:
        if conn:
            conn.execute("ROLLBACK")
        logger.error(f"Error cleaning old training data: {str(e)}")
        return 0
    finally:
        if conn:
            conn.close()

# Make sure these modules are available
import time
