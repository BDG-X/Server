from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import os
import json
import sqlite3
import schedule
import time
import threading
from datetime import datetime
import logging
import subprocess
import nltk
import stat  # Added for permission debugging
import sys  # For detailed error reporting

# Configure error and exception handling
import traceback
def handle_exception(exc_type, exc_value, exc_traceback):
    # Log the exception details to stderr and the log file
    logger = logging.getLogger()
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Set global exception handler
sys.excepthook = handle_exception

# Import database helpers with proper error handling
try:
    from utils.db_helpers import (
        init_db, 
        store_interactions, 
        store_uploaded_model,
        store_training_data,
        create_training_job,
        update_training_job_status,
        get_training_job,
        get_device_training_jobs
    )
    logging.info("Successfully imported database helpers")
except ImportError as e:
    logging.critical(f"Failed to import database helpers: {e}")
    # Continue without raising to allow app to partially initialize for diagnostics
    traceback.print_exc()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("backdoor_ai.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Set the base directory for persistent storage (support both Render and Koyeb)
if os.getenv("KOYEB_STORAGE_PATH"):
    # Koyeb persistent storage path
    BASE_DIR = os.getenv("KOYEB_STORAGE_PATH", "/var/koyeb/storage")
    logger.info(f"Using Koyeb persistent storage at {BASE_DIR}")
else:
    # Render persistent storage path (or local fallback)
    BASE_DIR = os.getenv("RENDER_DISK_PATH", "/opt/render/project")
    logger.info(f"Using Render persistent storage at {BASE_DIR}")

DB_PATH = os.path.join(BASE_DIR, "data", "interactions.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training_data")

# Create a thread lock for database operations
db_lock = threading.RLock()

# Ensure directories exist with proper error handling
try:
    for path in [os.path.dirname(DB_PATH), MODEL_DIR, TRAINING_DATA_DIR]:
        os.makedirs(path, exist_ok=True)
        if not os.path.exists(path):
            logger.critical(f"Failed to create directory: {path}")
        else:
            logger.info(f"Directory exists: {path}")
            # Check if it's writable
            if os.access(path, os.W_OK):
                logger.info(f"Directory is writable: {path}")
                # Try to write a test file
                try:
                    test_file = os.path.join(path, ".test_write")
                    with open(test_file, 'w') as f:
                        f.write("test")
                    os.remove(test_file)
                    logger.info(f"Successfully wrote test file in {path}")
                except Exception as e:
                    logger.error(f"Failed to write test file in {path}: {e}")
            else:
                logger.critical(f"Directory not writable: {path}")
except Exception as e:
    logger.critical(f"Error ensuring directories exist: {e}")
    logger.critical(traceback.format_exc())

# Debug: Check permissions of the base directory
def check_permissions(path):
    try:
        stats = os.stat(path)
        permissions = stat.filemode(stats.st_mode)
        logger.info(f"Permissions for {path}: {permissions}")
    except Exception as e:
        logger.error(f"Cannot check permissions for {path}: {e}")

check_permissions(BASE_DIR)

# Set NLTK data path to persistent disk
NLTK_DATA_PATH = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download NLTK data if not present
try:
    for resource in ['punkt', 'stopwords', 'wordnet']:
        resource_path = os.path.join(NLTK_DATA_PATH, resource)
        if not os.path.exists(resource_path):
            logger.info(f"Downloading NLTK resource: {resource} to {NLTK_DATA_PATH}")
            nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data to {NLTK_DATA_PATH}: {e}. Using fallback.")
    # Don't raise the exception, attempt to continue without these resources
    pass

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Get port from environment variable (Koyeb and Render both use PORT)
PORT = int(os.getenv("PORT", 8000))
logger.info(f"Server will run on port {PORT}")

# Initialize database with proper error handling
try:
    logger.info(f"Initializing database at {DB_PATH}")
    init_db(DB_PATH)
    logger.info("Database initialization successful")
except Exception as e:
    logger.critical(f"Database initialization failed: {e}")
    logger.critical(traceback.format_exc())
    # Continue execution to allow other endpoints to function
    # This allows the health check endpoint to report the actual error

@app.route('/api/ai/learn', methods=['POST'])
def collect_data():
    try:
        data = request.json
        logger.info(f"Received learning data from device: {data.get('deviceId', 'unknown')}")
        if not data or 'interactions' not in data:
            return jsonify({'success': False, 'message': 'Invalid data format'}), 400
        
        # Use thread lock for database operations
        with db_lock:
            store_interactions(DB_PATH, data)
            
        latest_model = get_latest_model_info()
        return jsonify({
            'success': True,
            'message': 'Data received successfully',
            'latestModelVersion': latest_model['version'],
            'modelDownloadURL': f"https://{request.host}/api/ai/models/{latest_model['version']}"
        })
    except Exception as e:
        logger.error(f"Error processing learning data: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/ai/upload-model', methods=['POST'])
def upload_model():
    """Endpoint for uploading user-trained .mlmodel files to be combined with server models."""
    
    try:
        # Check if file is included in the request
        if 'model' not in request.files:
            return jsonify({'success': False, 'message': 'No model file provided'}), 400
        
        model_file = request.files['model']
        
        # Check if a valid file was selected
        if model_file.filename == '':
            return jsonify({'success': False, 'message': 'No model file selected'}), 400
        
        # Get device ID and other metadata
        device_id = request.form.get('deviceId', 'unknown')
        app_version = request.form.get('appVersion', 'unknown')
        description = request.form.get('description', '')
        
        # Ensure the file is a CoreML model
        if not model_file.filename.endswith('.mlmodel'):
            return jsonify({'success': False, 'message': 'File must be a CoreML model (.mlmodel)'}), 400
        
        # Create directory for uploaded models if it doesn't exist
        UPLOADED_MODELS_DIR = os.path.join(MODEL_DIR, "uploaded")
        os.makedirs(UPLOADED_MODELS_DIR, exist_ok=True)
        
        # Generate a unique filename
        timestamp = int(datetime.now().timestamp())
        unique_filename = f"model_upload_{device_id}_{timestamp}.mlmodel"
        file_path = os.path.join(UPLOADED_MODELS_DIR, unique_filename)
        
        # Save the uploaded model
        model_file.save(file_path)
        logger.info(f"Saved uploaded model from device {device_id} to {file_path}")
        
        # Store model metadata in database
        with db_lock:
            model_id = store_uploaded_model(
                DB_PATH, 
                device_id=device_id,
                app_version=app_version,
                description=description,
                file_path=file_path,
                file_size=os.path.getsize(file_path),
                original_filename=model_file.filename
            )
        
        # Trigger async model retraining if enough models are uploaded
        from learning.trainer import should_retrain, trigger_retraining
        if should_retrain(DB_PATH):
            threading.Thread(target=trigger_retraining, args=(DB_PATH,), daemon=True).start()
            retraining_status = "Model retraining triggered"
        else:
            retraining_status = "Model will be incorporated in next scheduled training"
        
        # Return success response
        latest_model = get_latest_model_info()
        return jsonify({
            'success': True,
            'message': f'Model uploaded successfully. {retraining_status}',
            'modelId': model_id,
            'latestModelVersion': latest_model['version'],
            'modelDownloadURL': f"https://{request.host}/api/ai/models/{latest_model['version']}"
        })
        
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/api/ai/models/<version>', methods=['GET'])
def get_model(version):
    
    model_path = os.path.join(MODEL_DIR, f"model_{version}.mlmodel")
    if os.path.exists(model_path):
        logger.info(f"Serving model version {version}")
        try:
            return send_file(model_path, mimetype='application/octet-stream')
        except Exception as e:
            logger.error(f"Error sending model file: {str(e)}")
            return jsonify({'success': False, 'message': f'Error retrieving model: {str(e)}'}), 500
    else:
        logger.warning(f"Model version {version} not found")
        return jsonify({'success': False, 'message': 'Model not found'}), 404

# API endpoints for uploading models and training data are now handled through the web UI in the home page route

def get_available_model_versions():
    """Get a list of available model versions on disk"""
    versions = []
    
    try:
        # List all model files in the model directory
        model_files = [f for f in os.listdir(MODEL_DIR) 
                      if f.startswith('model_') and f.endswith('.mlmodel')]
        
        for file in model_files:
            # Extract version from filename
            match = re.search(r'model_([0-9.]+)\.mlmodel', file)
            if match:
                version = match.group(1)
                file_path = os.path.join(MODEL_DIR, file)
                file_size = os.path.getsize(file_path)
                modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                versions.append({
                    'version': version,
                    'path': file_path,
                    'size': file_size,
                    'modified': modified_time.isoformat()
                })
        
        # Sort by version (assuming semantic versioning)
        versions.sort(key=lambda v: [int(x) for x in v['version'].split('.')])
        
    except Exception as e:
        logger.error(f"Error listing model versions: {str(e)}")
    
    return versions

@app.route('/api/ai/latest-model', methods=['GET'])
def latest_model():
    model_info = get_latest_model_info()
    return jsonify({
        'success': True,
        'message': 'Latest model info',
        'latestModelVersion': model_info['version'],
        'modelDownloadURL': f"https://{request.host}/api/ai/models/{model_info['version']}"
    })

@app.route('/api/ai/stats', methods=['GET'])
def get_stats():
    
    conn = None
    try:
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT device_id) FROM interactions")
            unique_devices = cursor.fetchone()[0]
            cursor.execute("SELECT AVG(rating) FROM feedback")
            avg_rating = cursor.fetchone()[0] or 0
            cursor.execute("""
                SELECT detected_intent, COUNT(*) as count 
                FROM interactions 
                GROUP BY detected_intent 
                ORDER BY count DESC 
                LIMIT 5
            """)
            top_intents = [{"intent": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        model_info = get_latest_model_info()
        return jsonify({
            'success': True,
            'stats': {
                'totalInteractions': total_interactions,
                'uniqueDevices': unique_devices,
                'averageFeedbackRating': round(avg_rating, 2),
                'topIntents': top_intents,
                'latestModelVersion': model_info['version'],
                'lastTrainingDate': model_info.get('training_date', 'Unknown')
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500
    finally:
        if conn:
            conn.close()

def get_latest_model_info():
    """
    Get information about the latest model with enhanced error handling
    and security checks.
    
    Returns a dictionary with model information.
    """
    from utils.db_helpers import get_model_info
    
    try:
        # Try to get model info from the database first (most reliable source)
        model_info = get_model_info(DB_PATH)
        
        if model_info and 'version' in model_info:
            # Check if the model file actually exists
            if model_info.get('path') and os.path.exists(model_info['path']):
                return model_info
            else:
                logger.warning(f"Model file missing: {model_info.get('path')}")
                # Continue to fallback methods
        
        # Fallback to reading from the JSON file
        info_path = os.path.join(MODEL_DIR, "latest_model.json")
        
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    file_info = json.load(f)
                
                # Validate the model file exists
                if 'path' in file_info and os.path.exists(file_info['path']):
                    return file_info
                else:
                    # File path is invalid, but version info might still be useful
                    logger.warning(f"Model file referenced in JSON not found: {file_info.get('path')}")
                    model_path = os.path.join(MODEL_DIR, f"model_{file_info.get('version', '1.0.0')}.mlmodel")
                    file_info['path'] = model_path if os.path.exists(model_path) else None
                    return file_info
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading model info file: {str(e)}")
                # Continue to default fallback
        
        # Create default info if nothing else is available
        default_version = '1.0.0'
        default_path = os.path.join(MODEL_DIR, f"model_{default_version}.mlmodel")
        
        # Check if default model exists, update path accordingly
        if not os.path.exists(default_path):
            # Look for any model files in the directory
            model_files = [f for f in os.listdir(MODEL_DIR) if f.startswith('model_') and f.endswith('.mlmodel')]
            if model_files:
                # Use the newest model file based on modification time
                newest_model = max(model_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)))
                default_path = os.path.join(MODEL_DIR, newest_model)
                # Extract version from filename
                version_match = re.search(r'model_([0-9.]+)\.mlmodel', newest_model)
                if version_match:
                    default_version = version_match.group(1)
            else:
                # No models found
                default_path = None
                logger.warning("No model files found in model directory")
        
        default_info = {
            'version': default_version,
            'path': default_path,
            'training_date': datetime.now().isoformat()
        }
        
        # Create the model info file for future use
        try:
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            with open(info_path, 'w') as f:
                json.dump(default_info, f)
        except IOError as e:
            logger.error(f"Failed to write default model info: {str(e)}")
        
        return default_info
    
    except Exception as e:
        logger.error(f"Error accessing model info: {str(e)}", exc_info=True)
        # Last resort fallback
        return {
            'version': '1.0.0',
            'path': None,
            'training_date': datetime.now().isoformat(),
            'error': str(e)
        }

def train_model_job():
    # Lazy import to avoid circular import
    from learning.trainer import train_new_model
    try:
        logger.info("Starting scheduled model training")
        with db_lock:
            new_version = train_new_model(DB_PATH)
        logger.info(f"Model training completed. New version: {new_version}")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")

def cleanup_old_data():
    """
    Scheduled job to clean up old training data and temporary files
    to prevent disk space issues in long-running deployments
    """
    try:
        logger.info("Starting scheduled data cleanup")
        
        # Clean up old training data files
        from utils.db_helpers import clean_old_training_data
        deleted_count = clean_old_training_data(DB_PATH, days_old=30)
        
        # Clean up old log files (keep last 7 days)
        log_dir = os.path.dirname(os.path.abspath("backdoor_ai.log"))
        log_pattern = os.path.join(log_dir, "backdoor_ai*.log")
        try:
            import glob
            log_files = glob.glob(log_pattern)
            # Sort by modification time (oldest first)
            log_files.sort(key=lambda x: os.path.getmtime(x))
            
            # Keep the 10 most recent log files
            if len(log_files) > 10:
                for old_log in log_files[:-10]:
                    try:
                        os.remove(old_log)
                        logger.info(f"Deleted old log file: {old_log}")
                    except OSError as e:
                        logger.error(f"Error deleting log file {old_log}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up log files: {str(e)}")
        
        # Check disk space and log warning if low
        try:
            import shutil
            disk_usage = shutil.disk_usage(BASE_DIR)
            percent_free = (disk_usage.free / disk_usage.total) * 100
            if percent_free < 10:
                logger.warning(f"Low disk space: {percent_free:.1f}% free ({disk_usage.free / (1024*1024*1024):.2f} GB)")
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
            
        logger.info(f"Cleanup completed: removed {deleted_count} old training data files")
    except Exception as e:
        logger.error(f"Error in data cleanup job: {str(e)}")

def run_scheduler():
    """Run scheduled tasks with robust error handling"""
    logger.info("Initializing scheduler")
    
    try:
        # Schedule model training (early morning)
        schedule.every().day.at("02:00").do(train_model_job)
        logger.info("Scheduled daily model training for 02:00")
        
        # Schedule cleanup job (late night)
        schedule.every().day.at("03:30").do(cleanup_old_data)
        logger.info("Scheduled daily cleanup for 03:30")
        
        # Schedule disk space check (every 6 hours)
        def check_disk_space():
            try:
                import shutil
                disk_usage = shutil.disk_usage(BASE_DIR)
                percent_free = (disk_usage.free / disk_usage.total) * 100
                logger.info(f"Disk space check: {percent_free:.1f}% free ({disk_usage.free / (1024*1024*1024):.2f} GB)")
                if percent_free < 10:
                    logger.warning(f"Low disk space: {percent_free:.1f}% free ({disk_usage.free / (1024*1024*1024):.2f} GB)")
            except Exception as e:
                logger.error(f"Error checking disk space: {str(e)}")
        
        schedule.every(6).hours.do(check_disk_space)
        logger.info("Scheduled disk space check every 6 hours")
        
        # Run initial disk space check
        try:
            check_disk_space()
        except Exception as e:
            logger.error(f"Initial disk space check failed: {str(e)}")
        
        # Run scheduler loop with error handling
        logger.info("Starting scheduler loop")
        while True:
            try:
                schedule.run_pending()
            except Exception as e:
                logger.error(f"Error in scheduler: {str(e)}")
                logger.error(traceback.format_exc())
                # Don't exit the loop, just continue to the next iteration
            
            # Sleep with interruption handling
            try:
                time.sleep(60)
            except (KeyboardInterrupt, SystemExit):
                logger.info("Scheduler shutting down")
                break
            except Exception as e:
                logger.error(f"Error during scheduler sleep: {str(e)}")
                # Use a shorter sleep if there was an error
                time.sleep(5)
    
    except Exception as e:
        logger.critical(f"Fatal error in scheduler initialization: {str(e)}")
        logger.critical(traceback.format_exc())

# Start scheduler thread with error handling
try:
    logger.info("Starting scheduler thread")
    scheduler_thread = threading.Thread(target=run_scheduler, name="SchedulerThread")
    scheduler_thread.daemon = True
    scheduler_thread.start()
    logger.info("Scheduler thread started successfully")
except Exception as e:
    logger.critical(f"Failed to start scheduler thread: {str(e)}")
    logger.critical(traceback.format_exc())

# Log startup event with system info
logger.info(f"Started scheduler with daily model training at 02:00 and cleanup at 03:30")

# Add comprehensive health check endpoint for monitoring
@app.route('/health', methods=['GET'])
def health_check():
    """
    Comprehensive health check endpoint that provides detailed system status
    information for monitoring and diagnostics.
    
    Returns:
        JSON with health status of various system components
    """
    start_time = datetime.now()
    health_data = {
        'status': 'up',
        'timestamp': start_time.isoformat(),
        'components': {},
        'metrics': {}
    }
    
    # Always include detailed diagnostics (no key required)
    include_details = True
    
    try:
        # Get server uptime
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                health_data['uptime_seconds'] = uptime_seconds
                days, remainder = divmod(uptime_seconds, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                health_data['uptime_formatted'] = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"
        except Exception as e:
            logger.warning(f"Could not determine system uptime: {str(e)}")
        
        # 1. Database check
        db_component = {'status': 'unknown'}
        conn = None
        try:
            start = time.time()
            conn = sqlite3.connect(DB_PATH, timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master")
            table_count = cursor.fetchone()[0]
            conn.close()
            
            db_component.update({
                'status': 'healthy',
                'response_time_ms': int((time.time() - start) * 1000),
                'tables': table_count
            })
            
            # Add table counts if admin
            if include_details:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM interactions")
                db_component['interaction_count'] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM training_jobs")
                db_component['job_count'] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM model_versions")
                db_component['model_count'] = cursor.fetchone()[0]
                conn.close()
                
        except Exception as e:
            db_component.update({
                'status': 'unhealthy',
                'error': str(e)
            })
        finally:
            if conn and not conn.closed:
                conn.close()
        
        health_data['components']['database'] = db_component
        
        # 2. Model file storage
        models_component = {'status': 'unknown'}
        try:
            # Check if directory exists and is accessible
            if os.path.exists(MODEL_DIR) and os.access(MODEL_DIR, os.R_OK | os.W_OK):
                # Count models
                model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.mlmodel')]
                latest_model = get_latest_model_info()
                
                models_component.update({
                    'status': 'healthy',
                    'model_count': len(model_files),
                    'latest_version': latest_model.get('version'),
                    'directory': MODEL_DIR
                })
                
                # Verify latest model file exists
                if latest_model.get('path') and os.path.exists(latest_model.get('path')):
                    latest_model_size = os.path.getsize(latest_model.get('path'))
                    models_component['latest_model_size_mb'] = round(latest_model_size / (1024 * 1024), 2)
                else:
                    models_component['warning'] = "Latest model file missing"
            else:
                models_component.update({
                    'status': 'unhealthy',
                    'error': "Model directory not accessible"
                })
        except Exception as e:
            models_component.update({
                'status': 'unhealthy',
                'error': str(e)
            })
            
        health_data['components']['models'] = models_component
        
        # 3. System metrics
        try:
            import psutil
            import shutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_info = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_percent': memory.percent
            }
            
            # Disk usage
            disk_usage = shutil.disk_usage(BASE_DIR)
            disk_info = {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'used_percent': round((disk_usage.used / disk_usage.total) * 100, 2)
            }
            
            health_data['metrics'] = {
                'cpu_percent': cpu_percent,
                'memory': memory_info,
                'disk': disk_info
            }
            
            # Add warning flags
            health_data['warnings'] = []
            if cpu_percent > 90:
                health_data['warnings'].append("High CPU usage")
            if memory.percent > 90:
                health_data['warnings'].append("High memory usage")
            if (disk_usage.free / disk_usage.total) < 0.1:
                health_data['warnings'].append("Low disk space")
                
        except ImportError:
            health_data['metrics'] = {
                'status': 'limited',
                'message': "psutil not installed, limited metrics available"
            }
            
            # Fallback to basic disk check
            try:
                import shutil
                disk_usage = shutil.disk_usage(BASE_DIR)
                health_data['metrics']['disk'] = {
                    'total_gb': round(disk_usage.total / (1024**3), 2),
                    'free_gb': round(disk_usage.free / (1024**3), 2),
                    'used_percent': round((disk_usage.used / disk_usage.total) * 100, 2)
                }
            except Exception as e:
                logger.warning(f"Could not get disk metrics: {str(e)}")
        except Exception as e:
            health_data['metrics'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # 4. Scheduler status
        scheduler_component = {'status': 'unknown'}
        try:
            next_jobs = []
            for job in schedule.jobs:
                next_run = job.next_run
                next_jobs.append({
                    'job': str(job),
                    'next_run': next_run.isoformat() if next_run else None,
                    'seconds_until_next_run': job.seconds_until_run() if next_run else None
                })
            
            scheduler_component.update({
                'status': 'healthy',
                'scheduled_jobs': len(schedule.jobs),
                'next_jobs': next_jobs
            })
        except Exception as e:
            scheduler_component.update({
                'status': 'unhealthy',
                'error': str(e)
            })
            
        health_data['components']['scheduler'] = scheduler_component
        
        # 5. Training jobs status
        jobs_component = {'status': 'unknown'}
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Get counts of jobs by status
            cursor.execute("SELECT status, COUNT(*) FROM training_jobs GROUP BY status")
            status_counts = {status: count for status, count in cursor.fetchall()}
            
            # Get active jobs
            cursor.execute("""
                SELECT id, status, progress, start_time, estimated_completion_time
                FROM training_jobs
                WHERE status = 'processing'
                ORDER BY start_time DESC
                LIMIT 3
            """)
            active_jobs = [dict(zip(['id', 'status', 'progress', 'start_time', 'estimated_completion_time'], row)) 
                          for row in cursor.fetchall()]
            
            conn.close()
            
            jobs_component.update({
                'status': 'healthy',
                'status_counts': status_counts,
                'active_jobs': active_jobs,
                'active_job_count': len(active_jobs)
            })
        except Exception as e:
            jobs_component.update({
                'status': 'unhealthy',
                'error': str(e)
            })
            
        health_data['components']['training_jobs'] = jobs_component
        
        # Calculate overall health
        component_statuses = [comp.get('status') for comp in health_data['components'].values()]
        if any(status == 'unhealthy' for status in component_statuses):
            health_data['status'] = 'degraded'
            # Find the unhealthy components
            unhealthy = [name for name, comp in health_data['components'].items() 
                        if comp.get('status') == 'unhealthy']
            health_data['degraded_components'] = unhealthy
            
        # Calculate response time
        health_data['response_time_ms'] = int((datetime.now() - start_time).total_seconds() * 1000)
            
        # Return appropriate status code
        if health_data['status'] == 'degraded':
            return jsonify(health_data), 200  # Still return 200 for monitoring tools
        else:
            return jsonify(health_data)
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# API Documentation page
@app.route('/', methods=['GET'])
def api_documentation():
    # HTML template with API docs and tabbed interface for uploading/training
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Backdoor AI API Documentation</title>
        <style>
            :root {
                --primary-color: #2563eb;
                --primary-hover: #1e40af;
                --secondary-color: #64748b;
                --bg-color: #f8fafc;
                --card-bg: #ffffff;
                --code-bg: #f1f5f9;
                --border-color: #e2e8f0;
                --text-color: #334155;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: var(--bg-color);
                margin: 0;
                padding: 20px;
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
            }
            
            header {
                margin-bottom: 20px;
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
            }
            
            /* Tab styles */
            .tabs {
                display: flex;
                list-style-type: none;
                margin: 0 0 20px 0;
                padding: 0;
                overflow: hidden;
                border-bottom: 1px solid var(--border-color);
                font-weight: 500;
            }
            
            .tab-item {
                cursor: pointer;
                padding: 12px 24px;
                transition: background-color 0.3s;
                border-bottom: 3px solid transparent;
                color: var(--secondary-color);
            }
            
            .tab-item:hover {
                background-color: rgba(37, 99, 235, 0.05);
                color: var(--primary-color);
            }
            
            .tab-item.active {
                color: var(--primary-color);
                border-bottom: 3px solid var(--primary-color);
                font-weight: 600;
            }
            
            .tab-content {
                display: none;
                padding: 20px 0;
                animation: fadeIn 0.5s;
            }
            
            .tab-content.active {
                display: block;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            /* Upload form styles */
            .upload-form {
                background-color: var(--card-bg);
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                padding: 25px;
                margin-bottom: 30px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
            }
            
            .file-input {
                width: 100%;
                padding: 10px;
                border: 1px dashed var(--border-color);
                border-radius: 6px;
                background-color: var(--bg-color);
                margin-bottom: 10px;
            }
            
            .file-info {
                display: none;
                padding: 10px;
                margin-top: 10px;
                background-color: var(--code-bg);
                border-radius: 4px;
                font-size: 14px;
            }
            
            .submit-btn {
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            
            .submit-btn:hover {
                background-color: var(--primary-hover);
            }
            
            .submit-btn:disabled {
                background-color: var(--secondary-color);
                cursor: not-allowed;
                opacity: 0.7;
            }
            
            /* Progress indicator styles */
            .progress-container {
                display: none;
                margin-top: 30px;
                padding: 20px;
                background-color: var(--card-bg);
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }
            
            .progress-bar {
                height: 10px;
                background-color: var(--code-bg);
                border-radius: 5px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .progress-fill {
                height: 100%;
                background-color: var(--primary-color);
                width: 0%;
                transition: width 0.5s;
            }
            
            .status-message {
                margin-top: 10px;
                font-size: 14px;
            }
            
            .result-container {
                display: none;
                margin-top: 30px;
                padding: 20px;
                background-color: var(--card-bg);
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                text-align: center;
            }
            
            .download-btn {
                display: inline-block;
                background-color: var(--primary-color);
                color: white;
                text-decoration: none;
                padding: 10px 20px;
                border-radius: 4px;
                margin-top: 15px;
                transition: background-color 0.2s;
            }
            
            .download-btn:hover {
                background-color: var(--primary-hover);
            }
            
            h1 {
                color: var(--primary-color);
                margin-bottom: 10px;
            }
            
            h2 {
                margin-top: 40px;
                padding-bottom: 10px;
                border-bottom: 1px solid var(--border-color);
            }
            
            h3 {
                margin-top: 25px;
                color: var(--secondary-color);
            }
            
            .endpoint-card {
                background-color: var(--card-bg);
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                padding: 20px;
                margin-bottom: 30px;
                position: relative;
            }
            
            .method {
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
                color: white;
                margin-right: 10px;
            }
            
            .get {
                background-color: #22c55e;
            }
            
            .post {
                background-color: #3b82f6;
            }
            
            .path {
                font-family: monospace;
                font-size: 18px;
                font-weight: 600;
                vertical-align: middle;
            }
            
            .copy-btn {
                position: absolute;
                top: 20px;
                right: 20px;
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.2s;
            }
            
            .copy-btn:hover {
                background-color: var(--primary-hover);
            }
            
            pre {
                background-color: var(--code-bg);
                padding: 15px;
                border-radius: 6px;
                overflow: auto;
                font-family: monospace;
                font-size: 14px;
            }
            
            code {
                font-family: monospace;
                background-color: var(--code-bg);
                padding: 2px 5px;
                border-radius: 4px;
                font-size: 14px;
            }
            
            .description {
                margin: 15px 0;
            }
            
            .auth-info {
                margin-top: 15px;
                padding: 10px;
                background-color: #fffbeb;
                border-left: 4px solid #f59e0b;
                border-radius: 4px;
            }
            
            .request-example, .response-example {
                margin-top: 15px;
            }
            
            .parameters {
                margin-top: 15px;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            
            th, td {
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid var(--border-color);
            }
            
            th {
                background-color: var(--code-bg);
                font-weight: 600;
            }
            
            footer {
                margin-top: 60px;
                text-align: center;
                padding-top: 20px;
                border-top: 1px solid var(--border-color);
                color: var(--secondary-color);
                font-size: 14px;
            }
            
            .tooltip {
                position: relative;
                display: inline-block;
            }
            
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 140px;
                background-color: #555;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 150%;
                left: 50%;
                margin-left: -75px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            
            .tooltip .tooltiptext::after {
                content: "";
                position: absolute;
                top: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: #555 transparent transparent transparent;
            }
            
            .operation-tabs {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 20px;
                margin-bottom: 20px;
            }
            
            .operation-tab {
                padding: 8px 16px;
                background-color: var(--code-bg);
                border-radius: 4px;
                cursor: pointer;
                font-weight: 500;
                transition: background-color 0.2s;
            }
            
            .operation-tab:hover {
                background-color: #e2e8f0;
            }
            
            .operation-tab.active {
                background-color: var(--primary-color);
                color: white;
            }
            
            .operation-content {
                display: none;
                margin-top: 20px;
            }
            
            .operation-content.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>Backdoor AI Learning Server</h1>
                <p>Documentation and Training Interface</p>
            </header>
            
            <!-- Tab Navigation -->
            <ul class="tabs">
                <li class="tab-item active" onclick="openTab(event, 'endpoints-tab')">API Endpoints</li>
                <li class="tab-item" onclick="openTab(event, 'upload-train-tab')">Upload & Train</li>
            </ul>
            
            <!-- API Endpoints Tab Content -->
            <div id="endpoints-tab" class="tab-content active">
                <h2>Endpoints</h2>
            
            <!-- ALL-IN-ONE /api/ai/model-services -->
            <div class="endpoint-card">
                <span class="method post">POST</span>
                <span class="method get">GET</span>
                <span class="path">/api/ai/model-services</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/model-services')">Copy URL</button>
                
                <div class="description">
                    <p>A unified endpoint for all model-related operations including: uploading models, uploading training data, triggering training, checking status, and downloading models.</p>
                    <p>The operation is determined by the <code>operation</code> query parameter.</p>
                </div>
                
                <div class="auth-info">
                    <strong>No Authentication Required:</strong> This endpoint is publicly accessible.
                </div>
                
                <div class="operation-tabs">
                    <div class="operation-tab active" onclick="showOperation(event, 'upload-model')">Upload Model</div>
                    <div class="operation-tab" onclick="showOperation(event, 'upload-data')">Upload Training Data</div>
                    <div class="operation-tab" onclick="showOperation(event, 'train')">Trigger Training</div>
                    <div class="operation-tab" onclick="showOperation(event, 'status')">Check Status</div>
                    <div class="operation-tab" onclick="showOperation(event, 'download')">Download Model</div>
                </div>
                
                <!-- Upload Model Operation -->
                <div id="upload-model" class="operation-content active">
                    <h3>Upload Model</h3>
                    <p>Upload a CoreML model trained on your device to be incorporated into an ensemble.</p>
                    
                    <div class="parameters">
                        <h4>Request Format</h4>
                        <p><strong>Method:</strong> POST</p>
                        <p><strong>URL:</strong> <code>/api/ai/model-services?operation=upload-model</code></p>
                        <p>This operation requires a <code>multipart/form-data</code> request with the following fields:</p>
                        <table>
                            <tr>
                                <th>Field</th>
                                <th>Type</th>
                                <th>Description</th>
                            </tr>
                            <tr>
                                <td>model</td>
                                <td>File</td>
                                <td>The CoreML (.mlmodel) file to upload</td>
                            </tr>
                            <tr>
                                <td>deviceId</td>
                                <td>String</td>
                                <td>The unique identifier of the uploading device</td>
                            </tr>
                            <tr>
                                <td>appVersion</td>
                                <td>String</td>
                                <td>The version of the app sending the model</td>
                            </tr>
                            <tr>
                                <td>description</td>
                                <td>String</td>
                                <td>Optional description of the model</td>
                            </tr>
                            <tr>
                                <td>trainNow</td>
                                <td>Boolean</td>
                                <td>If 'true', immediately trains a new model using this upload; default is 'false'</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="response-example">
                        <h4>Response Example</h4>
                        <pre>{
  "success": true,
  "message": "Model uploaded successfully",
  "modelId": "d290f1ee-6c54-4b01-90e6-d701748f0851",
  "training": {
    "jobId": "f8a7b6c5-9e4d-3c2b-1a0f-8e7d6c5b4a3",
    "status": "queued",
    "estimatedTimeMinutes": 7,
    "message": "Training job started"
  },
  "latestModelVersion": "1.0.1712052481",
  "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
}</pre>
                    </div>
                </div>
                
                <!-- Upload Data Operation -->
                <div id="upload-data" class="operation-content">
                    <h3>Upload Training Data</h3>
                    <p>Upload JSON training data to be used for model training.</p>
                    
                    <div class="parameters">
                        <h4>Request Format</h4>
                        <p><strong>Method:</strong> POST</p>
                        <p><strong>URL:</strong> <code>/api/ai/model-services?operation=upload-data</code></p>
                        <p>This operation supports two content types:</p>
                        <ul>
                            <li><code>application/json</code>: Direct JSON in the request body</li>
                            <li><code>multipart/form-data</code>: JSON file upload with the form field "data"</li>
                        </ul>
                        <p>Additional query parameters or form fields:</p>
                        <table>
                            <tr>
                                <th>Field</th>
                                <th>Type</th>
                                <th>Description</th>
                            </tr>
                            <tr>
                                <td>deviceId</td>
                                <td>String</td>
                                <td>The unique identifier of the uploading device</td>
                            </tr>
                            <tr>
                                <td>appVersion</td>
                                <td>String</td>
                                <td>The version of the app sending the data</td>
                            </tr>
                            <tr>
                                <td>description</td>
                                <td>String</td>
                                <td>Optional description of the data</td>
                            </tr>
                            <tr>
                                <td>trainNow</td>
                                <td>Boolean</td>
                                <td>If 'true', immediately trains a new model using this data; default is 'false'</td>
                            </tr>
                            <tr>
                                <td>storeInteractions</td>
                                <td>Boolean</td>
                                <td>If 'true', also stores the interactions in the database; default is 'true'</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="request-example">
                        <h4>Request Body Example (application/json)</h4>
                        <pre>{
  "deviceId": "device_123",
  "appVersion": "1.2.0",
  "modelVersion": "1.0.0",
  "osVersion": "iOS 15.0",
  "interactions": [
    {
      "id": "int_abc123",
      "timestamp": "2023-06-15T14:30:00Z",
      "userMessage": "Turn on the lights",
      "aiResponse": "Turning on the lights",
      "detectedIntent": "light_on",
      "confidenceScore": 0.92,
      "feedback": {
        "rating": 5,
        "comment": "Perfect response"
      }
    }
  ]
}</pre>
                    </div>
                    
                    <div class="response-example">
                        <h4>Response Example</h4>
                        <pre>{
  "success": true,
  "message": "Training data uploaded successfully",
  "dataId": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
  "training": {
    "jobId": "f8a7b6c5-9e4d-3c2b-1a0f-8e7d6c5b4a3",
    "status": "queued",
    "estimatedTimeMinutes": 10,
    "message": "Training job started with 150 interactions"
  },
  "latestModelVersion": "1.0.1712052481",
  "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
}</pre>
                    </div>
                </div>
                
                <!-- Train Operation -->
                <div id="train" class="operation-content">
                    <h3>Trigger Training</h3>
                    <p>Trigger a model training job using existing data or models.</p>
                    
                    <div class="parameters">
                        <h4>Request Format</h4>
                        <p><strong>Method:</strong> POST</p>
                        <p><strong>URL:</strong> <code>/api/ai/model-services?operation=train</code></p>
                        <p>Available query parameters:</p>
                        <table>
                            <tr>
                                <th>Parameter</th>
                                <th>Type</th>
                                <th>Description</th>
                            </tr>
                            <tr>
                                <td>deviceId</td>
                                <td>String</td>
                                <td>The unique identifier of the requesting device</td>
                            </tr>
                            <tr>
                                <td>modelId</td>
                                <td>String</td>
                                <td>Optional ID of a specific model to use in training</td>
                            </tr>
                            <tr>
                                <td>dataId</td>
                                <td>String</td>
                                <td>Optional ID of specific training data to use</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="response-example">
                        <h4>Response Example</h4>
                        <pre>{
  "success": true,
  "message": "Training job started",
  "jobId": "f8a7b6c5-9e4d-3c2b-1a0f-8e7d6c5b4a3",
  "status": "queued",
  "estimatedTimeMinutes": 8
}</pre>
                    </div>
                </div>
                
                <!-- Status Operation -->
                <div id="status" class="operation-content">
                    <h3>Check Training Status</h3>
                    <p>Check the status of a training job or get all training jobs for a device.</p>
                    
                    <div class="parameters">
                        <h4>Request Format</h4>
                        <p><strong>Method:</strong> GET</p>
                        <p><strong>URL:</strong> <code>/api/ai/model-services?operation=status&amp;[jobId|deviceId]={id}</code></p>
                        <p>Required query parameters (one of):</p>
                        <table>
                            <tr>
                                <th>Parameter</th>
                                <th>Type</th>
                                <th>Description</th>
                            </tr>
                            <tr>
                                <td>jobId</td>
                                <td>String</td>
                                <td>ID of the specific training job to check</td>
                            </tr>
                            <tr>
                                <td>deviceId</td>
                                <td>String</td>
                                <td>Device ID to get all training jobs for that device</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="response-example">
                        <h4>Response Example (Specific Job)</h4>
                        <pre>{
  "success": true,
  "job": {
    "id": "f8a7b6c5-9e4d-3c2b-1a0f-8e7d6c5b4a3",
    "status": "processing",
    "progress": 0.75,
    "createdAt": "2025-04-03T14:30:00Z",
    "startTime": "2025-04-03T14:30:05Z",
    "estimatedCompletionTime": "2025-04-03T14:38:00Z",
    "actualCompletionTime": null
  }
}</pre>
                    </div>
                    
                    <div class="response-example">
                        <h4>Response Example (Device Jobs)</h4>
                        <pre>{
  "success": true,
  "jobs": [
    {
      "id": "f8a7b6c5-9e4d-3c2b-1a0f-8e7d6c5b4a3",
      "status": "completed",
      "progress": 1.0,
      "createdAt": "2025-04-03T14:30:00Z",
      "startTime": "2025-04-03T14:30:05Z",
      "estimatedCompletionTime": "2025-04-03T14:38:00Z",
      "actualCompletionTime": "2025-04-03T14:37:23Z",
      "resultModelVersion": "1.0.1712052489",
      "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052489"
    },
    {
      "id": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
      "status": "failed",
      "progress": 0.35,
      "createdAt": "2025-04-02T10:15:00Z",
      "startTime": "2025-04-02T10:15:05Z",
      "estimatedCompletionTime": "2025-04-02T10:23:00Z",
      "actualCompletionTime": "2025-04-02T10:17:42Z",
      "errorMessage": "Not enough data to train a model"
    }
  ]
}</pre>
                    </div>
                </div>
                
                <!-- Download Operation -->
                <div id="download" class="operation-content">
                    <h3>Download Model</h3>
                    <p>Download a trained model in CoreML format.</p>
                    
                    <div class="parameters">
                        <h4>Request Format</h4>
                        <p><strong>Method:</strong> GET</p>
                        <p><strong>URL:</strong> <code>/api/ai/model-services?operation=download&amp;version={version}</code></p>
                        <p>Available query parameters:</p>
                        <table>
                            <tr>
                                <th>Parameter</th>
                                <th>Type</th>
                                <th>Description</th>
                            </tr>
                            <tr>
                                <td>version</td>
                                <td>String</td>
                                <td>Optional model version to download; if not specified, downloads the latest version</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="response-example">
                        <h4>Response</h4>
                        <p>Binary file (CoreML model) with <code>application/octet-stream</code> content type, or error message if model not found.</p>
                    </div>
                </div>
            </div>
            
            <!-- POST /api/ai/learn -->
            <div class="endpoint-card">
                <span class="method post">POST</span>
                <span class="path">/api/ai/learn</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/learn')">Copy URL</button>
                
                <div class="description">
                    <p>Submit interaction data from devices to be used for model training. Returns information about the latest model version.</p>
                </div>
                
                <div class="auth-info">
                    <strong>No Authentication Required:</strong> This endpoint is publicly accessible.
                </div>
                
                <div class="request-example">
                    <h3>Request Example</h3>
                    <pre>{
  "deviceId": "device_123",
  "appVersion": "1.2.0",
  "modelVersion": "1.0.0",
  "osVersion": "iOS 15.0",
  "interactions": [
    {
      "id": "int_abc123",
      "timestamp": "2023-06-15T14:30:00Z",
      "userMessage": "Turn on the lights",
      "aiResponse": "Turning on the lights",
      "detectedIntent": "light_on",
      "confidenceScore": 0.92,
      "feedback": {
        "rating": 5,
        "comment": "Perfect response"
      }
    }
  ]
}</pre>
                </div>
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "success": true,
  "message": "Data received successfully",
  "latestModelVersion": "1.0.1712052481",
  "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
}</pre>
                </div>
            </div>
            
            <!-- POST /api/ai/upload-model -->
            <div class="endpoint-card">
                <span class="method post">POST</span>
                <span class="path">/api/ai/upload-model</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/upload-model')">Copy URL</button>
                
                <div class="description">
                    <p>Upload a CoreML model trained on your device to be combined with other models on the server. The server will create an ensemble model incorporating multiple uploaded models.</p>
                </div>
                
                <div class="auth-info">
                    <strong>No Authentication Required:</strong> This endpoint is publicly accessible.
                </div>
                
                <div class="request-example">
                    <h3>Request Format</h3>
                    <p>This endpoint requires a <code>multipart/form-data</code> request with the following fields:</p>
                    <table>
                        <tr>
                            <th>Field</th>
                            <th>Type</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td>model</td>
                            <td>File</td>
                            <td>The CoreML (.mlmodel) file to upload</td>
                        </tr>
                        <tr>
                            <td>deviceId</td>
                            <td>String</td>
                            <td>The unique identifier of the uploading device</td>
                        </tr>
                        <tr>
                            <td>appVersion</td>
                            <td>String</td>
                            <td>The version of the app sending the model</td>
                        </tr>
                        <tr>
                            <td>description</td>
                            <td>String</td>
                            <td>Optional description of the model</td>
                        </tr>
                    </table>
                </div>
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "success": true,
  "message": "Model uploaded successfully. Model will be incorporated in next scheduled training",
  "modelId": "d290f1ee-6c54-4b01-90e6-d701748f0851",
  "latestModelVersion": "1.0.1712052481",
  "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
}</pre>
                </div>
                
                <div class="description">
                    <h3>Model Processing</h3>
                    <p>After models are uploaded:</p>
                    <ul>
                        <li>They are stored on the server and queued for processing</li>
                        <li>When enough models are uploaded (3+) or after a time threshold, retraining is triggered</li>
                        <li>The server combines all uploaded models with its base model using ensemble techniques</li>
                        <li>The resulting model is available through the standard model endpoints</li>
                    </ul>
                </div>
            </div>
            
            <!-- GET /api/ai/models/{version} -->
            <div class="endpoint-card">
                <span class="method get">GET</span>
                <span class="path">/api/ai/models/{version}</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/models/1.0.0')">Copy URL</button>
                
                <div class="description">
                    <p>Download a specific model version. Returns the CoreML model file.</p>
                </div>
                
                <div class="auth-info">
                    <strong>No Authentication Required:</strong> This endpoint is publicly accessible.
                </div>
                
                <div class="parameters">
                    <h3>URL Parameters</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td>version</td>
                            <td>The version of the model to download (e.g., "1.0.0")</td>
                        </tr>
                    </table>
                </div>
                
                <div class="response-example">
                    <h3>Response</h3>
                    <p>Binary file (CoreML model) or error message if model not found.</p>
                </div>
            </div>
            
            <!-- GET /api/ai/latest-model -->
            <div class="endpoint-card">
                <span class="method get">GET</span>
                <span class="path">/api/ai/latest-model</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/latest-model')">Copy URL</button>
                
                <div class="description">
                    <p>Get information about the latest trained model. Returns the version and download URL.</p>
                </div>
                
                <div class="auth-info">
                    <strong>No Authentication Required:</strong> This endpoint is publicly accessible.
                </div>
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "success": true,
  "message": "Latest model info",
  "latestModelVersion": "1.0.1712052481",
  "modelDownloadURL": "https://yourdomain.com/api/ai/models/1.0.1712052481"
}</pre>
                </div>
            </div>
            
            <!-- GET /api/ai/stats -->
            <div class="endpoint-card">
                <span class="method get">GET</span>
                <span class="path">/api/ai/stats</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/api/ai/stats')">Copy URL</button>
                
                <div class="description">
                    <p>Get statistics about the collected data and model training. For admin use only.</p>
                </div>
                
                <div class="auth-info">
                    <strong>No Authentication Required:</strong> This endpoint is publicly accessible.
                </div>
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "success": true,
  "stats": {
    "totalInteractions": 1250,
    "uniqueDevices": 48,
    "averageFeedbackRating": 4.32,
    "topIntents": [
      {"intent": "light_on", "count": 325},
      {"intent": "temperature_query", "count": 214},
      {"intent": "music_play", "count": 186},
      {"intent": "weather_query", "count": 142},
      {"intent": "timer_set", "count": 95}
    ],
    "latestModelVersion": "1.0.1712052481",
    "lastTrainingDate": "2025-04-01T02:00:00Z"
  }
}</pre>
                </div>
            </div>
            
            <!-- GET /health -->
            <div class="endpoint-card">
                <span class="method get">GET</span>
                <span class="path">/health</span>
                <button class="copy-btn" onclick="copyToClipboard('https://' + window.location.host + '/health')">Copy URL</button>
                
                <div class="description">
                    <p>Health check endpoint to verify the server is running properly. Checks database and model storage accessibility.</p>
                </div>
                
                <div class="response-example">
                    <h3>Response Example</h3>
                    <pre>{
  "status": "up",
  "database": "healthy",
  "models": "healthy",
  "timestamp": "2025-04-01T10:15:30Z"
}</pre>
                </div>
            </div>
            
            <!-- Upload & Train Tab Content -->
            <div id="upload-train-tab" class="tab-content">
                <h2>Upload Model & Training Data</h2>
                <p>Use this interface to upload CoreML models and training data for the Backdoor AI server. You can upload either type of file or both.</p>
                
                <form id="upload-form" class="upload-form" onsubmit="submitTrainingForm(event)">
                    <!-- Device ID Field -->
                    <div class="form-group">
                        <label for="device-id">Device ID (optional)</label>
                        <input type="text" id="device-id" name="deviceId" placeholder="Enter your device ID" 
                               class="file-input" style="border: 1px solid var(--border-color);">
                    </div>
                    
                    <!-- Model File Upload -->
                    <div class="form-group">
                        <label for="model-file">Upload CoreML Model (.mlmodel)</label>
                        <input type="file" id="model-file" name="model" accept=".mlmodel" 
                               class="file-input" onchange="handleFileSelect('model-file', 'model-info')">
                        <div id="model-info" class="file-info"></div>
                        <p class="description">Upload a CoreML model file to be trained or combined with training data.</p>
                    </div>
                    
                    <!-- Training Data Upload -->
                    <div class="form-group">
                        <label for="data-file">Upload Training Data (.json)</label>
                        <input type="file" id="data-file" name="data" accept=".json" 
                               class="file-input" onchange="handleFileSelect('data-file', 'data-info')">
                        <div id="data-info" class="file-info"></div>
                        <p class="description">Upload JSON training data with interaction examples.</p>
                    </div>
                    
                    <!-- Submit Button -->
                    <button type="submit" id="submit-train-btn" class="submit-btn" disabled>
                        Upload & Start Training
                    </button>
                </form>
                
                <!-- Progress Container -->
                <div id="progress-container" class="progress-container">
                    <h3>Training Progress</h3>
                    <div class="progress-bar">
                        <div id="progress-fill" class="progress-fill"></div>
                    </div>
                    <p id="status-message" class="status-message">Preparing to train...</p>
                </div>
                
                <!-- Results Container -->
                <div id="result-container" class="result-container">
                    <h3>Training Complete!</h3>
                    <p>Your model has been successfully trained and is ready for download.</p>
                    <a id="download-link" href="#" class="download-btn">Download Model</a>
                </div>
            </div>
            
            <footer>
                <p>Backdoor AI Learning Server &copy; 2025</p>
            </footer>
        </div>
        
        <script>
            function copyToClipboard(text) {
                navigator.clipboard.writeText(text).then(function() {
                    var buttons = document.getElementsByClassName('copy-btn');
                    for (var i = 0; i < buttons.length; i++) {
                        buttons[i].textContent = 'Copy URL';
                    }
                    
                    var clickedButton = event.target;
                    var originalText = clickedButton.textContent;
                    clickedButton.textContent = 'Copied!';
                    
                    setTimeout(function() {
                        clickedButton.textContent = originalText;
                    }, 2000);
                }, function(err) {
                    console.error('Could not copy text: ', err);
                });
            }
            
            function showOperation(event, operationId) {
                // Hide all operation contents
                var contents = document.getElementsByClassName('operation-content');
                for (var i = 0; i < contents.length; i++) {
                    contents[i].classList.remove('active');
                }
                
                // Remove active class from all tabs
                var tabs = document.getElementsByClassName('operation-tab');
                for (var i = 0; i < tabs.length; i++) {
                    tabs[i].classList.remove('active');
                }
                
                // Show the selected content and activate the tab
                document.getElementById(operationId).classList.add('active');
                event.currentTarget.classList.add('active');
            }
            
            // Tab switching function
            function openTab(evt, tabId) {
                // Hide all tab content
                var tabContents = document.getElementsByClassName("tab-content");
                for (var i = 0; i < tabContents.length; i++) {
                    tabContents[i].classList.remove("active");
                }
                
                // Deactivate all tab buttons
                var tabLinks = document.getElementsByClassName("tab-item");
                for (var i = 0; i < tabLinks.length; i++) {
                    tabLinks[i].classList.remove("active");
                }
                
                // Show the selected tab content and activate tab
                document.getElementById(tabId).classList.add("active");
                evt.currentTarget.classList.add("active");
            }
            
            // File upload preview handler
            function handleFileSelect(inputId, infoId) {
                const fileInput = document.getElementById(inputId);
                const fileInfo = document.getElementById(infoId);
                
                if (fileInput.files.length > 0) {
                    const file = fileInput.files[0];
                    const fileSize = (file.size / 1024).toFixed(2) + " KB";
                    fileInfo.textContent = `Selected file: ${file.name} (${fileSize})`;
                    fileInfo.style.display = "block";
                } else {
                    fileInfo.style.display = "none";
                }
                
                // Enable/disable submit button based on both files being selected
                updateSubmitButton();
            }
            
            function updateSubmitButton() {
                const modelFile = document.getElementById('model-file').files.length > 0;
                const dataFile = document.getElementById('data-file').files.length > 0;
                const submitBtn = document.getElementById('submit-train-btn');
                
                // Enable button if at least one file is selected
                submitBtn.disabled = !(modelFile || dataFile);
            }
            
            // Form submission handler
            async function submitTrainingForm(event) {
                event.preventDefault();
                
                // Get form elements
                const modelFile = document.getElementById('model-file').files[0];
                const dataFile = document.getElementById('data-file').files[0];
                const deviceId = document.getElementById('device-id').value || 'web-client';
                
                // Ensure at least one file is selected
                if (!modelFile && !dataFile) {
                    alert('Please select at least one file to upload.');
                    return;
                }
                
                // Show progress container
                document.getElementById('progress-container').style.display = 'block';
                document.getElementById('upload-form').style.display = 'none';
                document.getElementById('status-message').textContent = 'Uploading files...';
                
                try {
                    let jobId = null;
                    let modelId = null;
                    let dataId = null;
                    
                    // Upload model file if selected
                    if (modelFile) {
                        const formData = new FormData();
                        formData.append('model', modelFile);
                        formData.append('deviceId', deviceId);
                        formData.append('trainNow', dataFile ? 'false' : 'true');
                        formData.append('description', 'Uploaded from web interface');
                        
                        const response = await fetch('/api/ai/upload-model', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (!result.success) {
                            throw new Error(result.message || 'Failed to upload model');
                        }
                        
                        modelId = result.modelId;
                        document.getElementById('status-message').textContent = 'Model uploaded successfully.';
                        
                        // If training was started and no data file, get job ID
                        if (result.training && result.training.jobId && !dataFile) {
                            jobId = result.training.jobId;
                        }
                    }
                    
                    // Upload data file if selected
                    if (dataFile) {
                        const formData = new FormData();
                        formData.append('data', dataFile);
                        formData.append('deviceId', deviceId);
                        formData.append('trainNow', modelFile ? 'false' : 'true');
                        formData.append('description', 'Uploaded from web interface');
                        
                        const response = await fetch('/api/ai/learn', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        
                        if (!result.success) {
                            throw new Error(result.message || 'Failed to upload training data');
                        }
                        
                        document.getElementById('status-message').textContent = 'Training data uploaded successfully.';
                    }
                    
                    // If both files were uploaded, trigger training
                    if (modelFile && dataFile && modelId) {
                        document.getElementById('status-message').textContent = 'Starting model training...';
                        
                        const trainResponse = await fetch('/api/ai/upload-model', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                deviceId: deviceId,
                                modelId: modelId,
                                trainNow: true
                            })
                        });
                        
                        const trainResult = await trainResponse.json();
                        
                        if (!trainResult.success) {
                            throw new Error(trainResult.message || 'Failed to start training');
                        }
                        
                        jobId = trainResult.training.jobId;
                    }
                    
                    // If we have a job ID, poll for progress
                    if (jobId) {
                        pollTrainingProgress(jobId);
                    } else {
                        // No job ID means no active training
                        document.getElementById('status-message').textContent = 'Files uploaded. Training will happen in the background.';
                        document.getElementById('progress-fill').style.width = '100%';
                    }
                    
                } catch (error) {
                    console.error('Error during upload/training:', error);
                    document.getElementById('status-message').textContent = 'Error: ' + error.message;
                }
            }
            
            // Poll for training job progress
            async function pollTrainingProgress(jobId) {
                try {
                    const response = await fetch(`/api/ai/latest-model`);
                    const result = await response.json();
                    
                    if (result.success) {
                        // Update progress UI
                        let progress = 1.0; // Completed
                        document.getElementById('progress-fill').style.width = `${progress * 100}%`;
                        document.getElementById('status-message').textContent = 'Training completed!';
                        
                        // Show result container with download link
                        const resultContainer = document.getElementById('result-container');
                        resultContainer.style.display = 'block';
                        
                        // Set download link
                        const downloadLink = document.getElementById('download-link');
                        downloadLink.href = result.modelDownloadURL;
                        downloadLink.download = `model_${result.latestModelVersion}.mlmodel`;
                        downloadLink.textContent = `Download Model (Version ${result.latestModelVersion})`;
                        
                        return; // Done polling
                    }
                    
                    // Continue polling
                    setTimeout(() => pollTrainingProgress(jobId), 5000);
                    
                } catch (error) {
                    console.error('Error polling training status:', error);
                    document.getElementById('status-message').textContent = 'Error checking training status: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    '''
    
    return render_template_string(html_template)

if __name__ == '__main__':
    pip_version = subprocess.check_output(["pip", "--version"]).decode("utf-8").strip()
    logger.info(f"Using pip version: {pip_version}")
    logger.info("Starting Backdoor AI Learning Server on Render")
    app.run(host='0.0.0.0', port=PORT, debug=False)