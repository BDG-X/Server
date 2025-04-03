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

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

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

# Initialize database
init_db(DB_PATH)

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

@app.route('/api/ai/model-services', methods=['GET', 'POST'])
def model_services():
    """
    Unified endpoint for model-related services, including:
    - Uploading models and training data
    - Triggering model training
    - Checking training status
    - Retrieving trained models
    
    Operations are determined by the 'operation' query parameter:
    - upload-model: Upload a CoreML model file
    - upload-data: Upload JSON training data
    - train: Trigger model training
    - status: Check training job status
    - download: Download a trained model
    """
    
    # Input validation: Sanitize and validate the operation parameter
    operation = request.args.get('operation', '')
    valid_operations = {'upload-model', 'upload-data', 'train', 'status', 'download'}
    
    if not operation:
        return jsonify({
            'success': False, 
            'message': 'Missing operation parameter',
            'validOperations': list(valid_operations)
        }), 400
        
    if operation not in valid_operations:
        return jsonify({
            'success': False, 
            'message': f'Invalid operation: {operation}',
            'validOperations': list(valid_operations)
        }), 400
    
    # Rate limiting (placeholder for actual implementation)
    # In production, you would implement proper rate limiting here
    
    try:
        # Security headers for file downloads
        headers = {}
        
        # ======== MODEL UPLOAD OPERATION ========
        if operation == 'upload-model' and request.method == 'POST':
            # File validation
            if 'model' not in request.files:
                return jsonify({'success': False, 'message': 'No model file provided'}), 400
            
            model_file = request.files['model']
            
            # Check file exists and has content
            if model_file.filename == '' or not model_file:
                return jsonify({'success': False, 'message': 'No model file selected or empty file uploaded'}), 400
            
            # Get and sanitize metadata
            from utils.db_helpers import sanitize_string
            device_id = sanitize_string(request.form.get('deviceId', 'unknown'), 100)
            app_version = sanitize_string(request.form.get('appVersion', 'unknown'), 50)
            description = sanitize_string(request.form.get('description', ''), 1000)
            
            # Convert trainNow to boolean safely 
            train_now = False
            train_now_input = request.form.get('trainNow', 'false').lower()
            if train_now_input in ('true', 'yes', '1', 'on'):
                train_now = True
            
            # Enhanced file validation
            filename = model_file.filename
            file_ext = os.path.splitext(filename)[1].lower() if filename else ''
            
            # Validate extension and MIME type
            if file_ext != '.mlmodel':
                return jsonify({'success': False, 'message': 'File must have .mlmodel extension'}), 400
            
            # Check mimetype if available 
            mimetype = model_file.content_type if hasattr(model_file, 'content_type') else None
            valid_mimetypes = {'application/octet-stream', 'application/x-mlmodel', ''}
            
            if mimetype and mimetype not in valid_mimetypes:
                logger.warning(f"Suspicious MIME type for model upload: {mimetype}")
                return jsonify({'success': False, 'message': 'Invalid file type'}), 400
            
            # Create secure directory for uploaded models
            UPLOADED_MODELS_DIR = os.path.join(MODEL_DIR, "uploaded")
            os.makedirs(UPLOADED_MODELS_DIR, exist_ok=True)
            
            # Generate a unique secure filename with UUID
            file_uuid = str(uuid.uuid4())
            timestamp = int(datetime.now().timestamp())
            unique_filename = f"model_upload_{device_id}_{timestamp}_{file_uuid}.mlmodel"
            file_path = os.path.join(UPLOADED_MODELS_DIR, unique_filename)
            
            # Check and create parent directory if needed (avoid path traversal)
            parent_dir = os.path.dirname(file_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Save the uploaded model safely with a try/except block
            try:
                model_file.save(file_path)
                logger.info(f"Saved uploaded model from device {device_id} to {file_path}")
                
                # Get file size with error handling
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size == 0:
                        raise ValueError("Uploaded file is empty")
                    
                    # Check if file is too large (e.g., 100MB limit)
                    max_size = 100 * 1024 * 1024  # 100MB
                    if file_size > max_size:
                        os.remove(file_path)
                        return jsonify({'success': False, 'message': 'File too large (max 100MB)'}), 413
                except OSError as e:
                    logger.error(f"Error accessing file: {str(e)}")
                    return jsonify({'success': False, 'message': 'Error processing uploaded file'}), 500
                
            except Exception as e:
                logger.error(f"Error saving uploaded model: {str(e)}")
                return jsonify({'success': False, 'message': 'Error saving uploaded file'}), 500
            
            model_id = None
            job_id = None
            
            # Use a database transaction to ensure consistency
            try:
                # Store model metadata in database
                with db_lock:
                    model_id = store_uploaded_model(
                        DB_PATH, 
                        device_id=device_id,
                        app_version=app_version,
                        description=description,
                        file_path=file_path,
                        file_size=file_size,
                        original_filename=os.path.basename(filename)
                    )
                
                # Create a training job if requested to train immediately
                if train_now and model_id:
                    # Create a training job
                    from learning.trainer import estimate_training_time
                    estimated_time = estimate_training_time(file_size, 1)
                    
                    # Prepare metadata
                    metadata = {
                        'source_filename': os.path.basename(filename),
                        'estimated_minutes': estimated_time,
                        'data_size': file_size,
                        'created_by': 'model_services_api'
                    }
                    
                    with db_lock:
                        job_id = create_training_job(
                            DB_PATH,
                            device_id=device_id,
                            source_type='uploaded_model',
                            source_id=model_id,
                            metadata=metadata
                        )
                    
                    # Start training in a background thread
                    from learning.trainer import trigger_retraining_job
                    if job_id:
                        training_thread = threading.Thread(
                            target=trigger_retraining_job,
                            args=(DB_PATH, job_id, 'uploaded_model', model_id),
                            daemon=True
                        )
                        training_thread.start()
                        
                        training_status = {
                            'jobId': job_id,
                            'status': 'queued',
                            'estimatedTimeMinutes': estimated_time,
                            'message': 'Training job started'
                        }
                    else:
                        training_status = {
                            'message': 'Failed to create training job',
                            'estimatedTimeMinutes': None
                        }
                else:
                    # If not training immediately, check if general retraining should be triggered
                    from learning.trainer import should_retrain, trigger_retraining
                    if should_retrain(DB_PATH):
                        threading.Thread(target=trigger_retraining, args=(DB_PATH,), daemon=True).start()
                        training_status = {
                            'message': 'Model will be incorporated in scheduled training (triggered)',
                            'estimatedTimeMinutes': None
                        }
                    else:
                        training_status = {
                            'message': 'Model will be incorporated in next scheduled training',
                            'estimatedTimeMinutes': None
                        }
                
                # Return success response with detailed information
                latest_model = get_latest_model_info()
                return jsonify({
                    'success': True,
                    'message': 'Model uploaded successfully',
                    'modelId': model_id,
                    'fileName': os.path.basename(filename),
                    'fileSize': file_size,
                    'uploadTime': datetime.now().isoformat(),
                    'training': training_status,
                    'latestModelVersion': latest_model['version'],
                    'modelDownloadURL': f"https://{request.host}/api/ai/models/{latest_model['version']}"
                })
                
            except Exception as e:
                # If anything fails during the database/training steps, clean up the file
                logger.error(f"Error in model upload process: {str(e)}")
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
                return jsonify({'success': False, 'message': f'Error processing model: {str(e)}'}), 500
        
        # ======== JSON DATA UPLOAD OPERATION ========
        elif operation == 'upload-data' and request.method == 'POST':
            from utils.db_helpers import sanitize_string
            
            # Get and sanitize metadata from form or query parameters
            device_id = sanitize_string(request.args.get('deviceId') or request.form.get('deviceId', 'unknown'), 100)
            app_version = sanitize_string(request.args.get('appVersion') or request.form.get('appVersion', 'unknown'), 50)
            description = sanitize_string(request.args.get('description') or request.form.get('description', ''), 1000)
            
            # Convert trainNow to boolean safely
            train_now_input = request.args.get('trainNow') or request.form.get('trainNow', 'false')
            train_now = train_now_input.lower() in ('true', 'yes', '1', 'on')
            
            # Convert storeInteractions to boolean safely
            store_interactions_input = request.args.get('storeInteractions') or request.form.get('storeInteractions', 'true')
            store_interactions_flag = store_interactions_input.lower() in ('true', 'yes', '1', 'on')
            
            # Determine content type
            content_type = request.content_type or ''
            data_content = None
            file_path = None
            file_size = 0
            
            # Create secure directory for uploaded data
            os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
            
            # Generate secure filenames with UUID
            file_uuid = str(uuid.uuid4())
            timestamp = int(datetime.now().timestamp())
            unique_filename = f"data_upload_{device_id}_{timestamp}_{file_uuid}.json"
            file_path = os.path.join(TRAINING_DATA_DIR, unique_filename)
            
            try:
                # Process direct JSON data
                if 'application/json' in content_type:
                    # JSON data provided directly in request body
                    data_content = request.json
                    
                    # Validate data structure
                    if not data_content or not isinstance(data_content, dict) or 'interactions' not in data_content:
                        return jsonify({'success': False, 'message': 'Invalid JSON data format: missing interactions array'}), 400
                    
                    if not isinstance(data_content.get('interactions', []), list):
                        return jsonify({'success': False, 'message': 'Invalid JSON data format: interactions must be an array'}), 400
                    
                    # Enforce size limits
                    interactions_count = len(data_content.get('interactions', []))
                    if interactions_count == 0:
                        return jsonify({'success': False, 'message': 'No interactions provided in data'}), 400
                    
                    if interactions_count > 50000:  # Arbitrary limit to prevent abuse
                        return jsonify({'success': False, 'message': 'Too many interactions (max 50,000)'}), 413
                    
                    # Save the JSON data to a file with proper error handling
                    try:
                        with open(file_path, 'w') as f:
                            json.dump(data_content, f)
                        
                        file_size = os.path.getsize(file_path)
                        logger.info(f"Saved JSON data from device {device_id} to {file_path} ({file_size} bytes)")
                    except (IOError, OSError) as e:
                        logger.error(f"Error saving JSON data: {str(e)}")
                        return jsonify({'success': False, 'message': 'Error saving training data'}), 500
                
                # Process file upload
                elif 'multipart/form-data' in content_type:
                    # File upload with form data
                    if 'data' not in request.files:
                        return jsonify({'success': False, 'message': 'No data file provided in form data'}), 400
                    
                    data_file = request.files['data']
                    
                    # Check if file exists and has content
                    if data_file.filename == '' or not data_file:
                        return jsonify({'success': False, 'message': 'No data file selected or empty file uploaded'}), 400
                    
                    # Validate file extension
                    orig_filename = data_file.filename
                    file_ext = os.path.splitext(orig_filename)[1].lower() if orig_filename else ''
                    
                    if file_ext != '.json':
                        return jsonify({'success': False, 'message': 'File must have .json extension'}), 400
                    
                    # Check file type if available
                    mimetype = data_file.content_type if hasattr(data_file, 'content_type') else None
                    valid_mimetypes = {'application/json', 'text/json', 'text/plain', ''}
                    
                    if mimetype and mimetype not in valid_mimetypes:
                        logger.warning(f"Suspicious MIME type for JSON upload: {mimetype}")
                        return jsonify({'success': False, 'message': 'Invalid file type'}), 400
                    
                    # Save the uploaded file with proper error handling
                    try:
                        data_file.save(file_path)
                        file_size = os.path.getsize(file_path)
                        
                        # Check file size
                        if file_size == 0:
                            os.remove(file_path)  # Clean up empty file
                            return jsonify({'success': False, 'message': 'Uploaded file is empty'}), 400
                        
                        # Maximum allowed file size (e.g., 10MB)
                        max_size = 10 * 1024 * 1024
                        if file_size > max_size:
                            os.remove(file_path)  # Clean up oversized file
                            return jsonify({'success': False, 'message': 'File too large (max 10MB)'}), 413
                        
                        logger.info(f"Saved JSON file from device {device_id} to {file_path} ({file_size} bytes)")
                    except Exception as e:
                        logger.error(f"Error saving uploaded file: {str(e)}")
                        if os.path.exists(file_path):
                            os.remove(file_path)  # Clean up on error
                        return jsonify({'success': False, 'message': 'Error saving uploaded file'}), 500
                    
                    # Load and validate the file content
                    try:
                        with open(file_path, 'r') as f:
                            data_content = json.load(f)
                        
                        # Validate data structure
                        if not data_content or not isinstance(data_content, dict) or 'interactions' not in data_content:
                            os.remove(file_path)  # Clean up invalid file
                            return jsonify({'success': False, 'message': 'Invalid JSON data format: missing interactions array'}), 400
                        
                        if not isinstance(data_content.get('interactions', []), list):
                            os.remove(file_path)  # Clean up invalid file
                            return jsonify({'success': False, 'message': 'Invalid JSON data format: interactions must be an array'}), 400
                        
                        interactions_count = len(data_content.get('interactions', []))
                        if interactions_count == 0:
                            os.remove(file_path)  # Clean up empty file
                            return jsonify({'success': False, 'message': 'No interactions provided in data'}), 400
                            
                    except json.JSONDecodeError as e:
                        os.remove(file_path)  # Clean up invalid file
                        return jsonify({'success': False, 'message': f'Invalid JSON file: {str(e)}'}), 400
                    except Exception as e:
                        if os.path.exists(file_path):
                            os.remove(file_path)  # Clean up on error
                        logger.error(f"Error validating uploaded JSON: {str(e)}")
                        return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'}), 500
                
                else:
                    # Invalid content type
                    return jsonify({
                        'success': False, 
                        'message': 'Invalid content type. Expected application/json or multipart/form-data',
                        'receivedContentType': content_type
                    }), 400
                
                # At this point we have valid data_content and file_path
                data_id = None
                job_id = None
                interactions_count = len(data_content.get('interactions', []))
                
                # Complete the database operations in a transaction
                try:
                    # Store training data metadata in database
                    with db_lock:
                        data_id = store_training_data(
                            DB_PATH,
                            device_id=device_id,
                            app_version=app_version,
                            description=description,
                            data_path=file_path,
                            file_size=file_size
                        )
                    
                    # Create a training job if requested
                    if train_now and data_id:
                        # Create a training job with appropriate time estimate
                        from learning.trainer import estimate_training_time
                        estimated_time = estimate_training_time(interactions_count)
                        
                        metadata = {
                            'interactions_count': interactions_count,
                            'estimated_minutes': estimated_time,
                            'data_size': file_size,
                            'created_by': 'model_services_api'
                        }
                        
                        with db_lock:
                            job_id = create_training_job(
                                DB_PATH,
                                device_id=device_id,
                                source_type='json_data',
                                source_id=data_id,
                                metadata=metadata
                            )
                        
                        # Start training in a background thread if job created successfully
                        if job_id:
                            from learning.trainer import trigger_retraining_job
                            training_thread = threading.Thread(
                                target=trigger_retraining_job,
                                args=(DB_PATH, job_id, 'json_data', data_id),
                                daemon=True
                            )
                            training_thread.start()
                            
                            training_status = {
                                'jobId': job_id,
                                'status': 'queued',
                                'estimatedTimeMinutes': estimated_time,
                                'message': f'Training job started with {interactions_count} interactions'
                            }
                        else:
                            training_status = {
                                'message': 'Failed to create training job',
                                'estimatedTimeMinutes': None
                            }
                    else:
                        training_status = {
                            'message': 'Data stored for next scheduled training',
                            'estimatedTimeMinutes': None
                        }
                    
                    # Store the interactions in the database (optional)
                    stored_count = 0
                    if store_interactions_flag and data_content and 'interactions' in data_content:
                        with db_lock:
                            try:
                                stored_count = store_interactions(DB_PATH, data_content)
                                logger.info(f"Stored {stored_count} interactions from uploaded data (device: {device_id})")
                            except Exception as e:
                                logger.error(f"Error storing interactions from uploaded data: {str(e)}")
                    
                    # Return success response with detailed information
                    latest_model = get_latest_model_info()
                    return jsonify({
                        'success': True,
                        'message': 'Training data uploaded successfully',
                        'dataId': data_id,
                        'interactionsCount': interactions_count,
                        'interactionsStored': stored_count,
                        'fileSize': file_size,
                        'uploadTime': datetime.now().isoformat(),
                        'training': training_status,
                        'latestModelVersion': latest_model['version'],
                        'modelDownloadURL': f"https://{request.host}/api/ai/models/{latest_model['version']}"
                    })
                    
                except Exception as e:
                    # If any database operation fails, clean up the file
                    logger.error(f"Error processing training data: {str(e)}")
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except:
                        pass
                    return jsonify({'success': False, 'message': f'Error processing training data: {str(e)}'}), 500
                    
            except Exception as e:
                logger.error(f"Unexpected error in data upload: {str(e)}", exc_info=True)
                # Clean up any partial file
                try:
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
                return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500
        
        # ======== TRIGGER TRAINING OPERATION ========
        elif operation == 'train' and request.method == 'POST':
            from utils.db_helpers import sanitize_string, validate_uuid
            
            # Get and validate parameters
            device_id = sanitize_string(request.args.get('deviceId') or request.form.get('deviceId', 'unknown'), 100)
            model_id = request.args.get('modelId') or request.form.get('modelId')
            data_id = request.args.get('dataId') or request.form.get('dataId')
            
            # Validate UUIDs if provided
            if model_id and not validate_uuid(model_id):
                return jsonify({'success': False, 'message': f'Invalid model ID format: {model_id}'}), 400
                
            if data_id and not validate_uuid(data_id):
                return jsonify({'success': False, 'message': f'Invalid data ID format: {data_id}'}), 400
            
            # Determine the source type based on provided IDs
            source_type = None
            source_id = None
            
            if model_id and data_id:
                source_type = 'combined'
                source_id = f"{model_id}_{data_id}"
            elif model_id:
                source_type = 'uploaded_model'
                source_id = model_id
            elif data_id:
                source_type = 'json_data'
                source_id = data_id
            else:
                source_type = 'scheduled'
                source_id = 'scheduled_training'
            
            # Verify resources exist if IDs provided
            if source_type != 'scheduled':
                if source_type == 'uploaded_model' or source_type == 'combined':
                    # Verify model exists
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT id, file_path FROM uploaded_models WHERE id = ?", (model_id,))
                    model_record = cursor.fetchone()
                    
                    if not model_record:
                        conn.close()
                        return jsonify({'success': False, 'message': f'Model not found: {model_id}'}), 404
                    
                    # Verify model file exists
                    if not os.path.exists(model_record[1]):
                        conn.close()
                        return jsonify({'success': False, 'message': f'Model file missing for ID: {model_id}'}), 404
                    
                if source_type == 'json_data' or source_type == 'combined':
                    # Verify training data exists
                    if source_type != 'combined':  # Only execute if not already connected
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        
                    cursor.execute("SELECT id, data_path FROM training_data WHERE id = ?", (data_id,))
                    data_record = cursor.fetchone()
                    conn.close()
                    
                    if not data_record:
                        return jsonify({'success': False, 'message': f'Training data not found: {data_id}'}), 404
                    
                    # Verify data file exists
                    if not os.path.exists(data_record[1]):
                        return jsonify({'success': False, 'message': f'Training data file missing for ID: {data_id}'}), 404
            
            # Create appropriate metadata for time estimation
            metadata = {'created_by': 'model_services_api_train_endpoint'}
            estimated_time = 5  # Default estimate
            
            # Get better time estimate based on source type
            from learning.trainer import estimate_training_time
            
            try:
                if source_type == 'json_data':
                    # Estimate based on number of interactions
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT data_path FROM training_data WHERE id = ?", (data_id,))
                    data_path_result = cursor.fetchone()
                    conn.close()
                    
                    if data_path_result and data_path_result[0] and os.path.exists(data_path_result[0]):
                        try:
                            with open(data_path_result[0], 'r') as f:
                                data_content = json.load(f)
                                interactions_count = len(data_content.get('interactions', []))
                                # Get file size for more accurate estimate
                                file_size = os.path.getsize(data_path_result[0])
                                
                                estimated_time = estimate_training_time(interactions_count)
                                metadata['interactions_count'] = interactions_count
                                metadata['data_size'] = file_size
                        except Exception as e:
                            logger.warning(f"Error reading training data for estimation: {str(e)}")
                
                elif source_type == 'uploaded_model':
                    # Estimate based on model complexity
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT file_size FROM uploaded_models WHERE id = ?", (model_id,))
                    file_size_result = cursor.fetchone()
                    conn.close()
                    
                    if file_size_result and file_size_result[0]:
                        model_size = file_size_result[0]
                        estimated_time = estimate_training_time(0, 1)  # Base on single model
                        metadata['model_size'] = model_size
                
                elif source_type == 'combined':
                    # Estimate based on both data and model
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    
                    # Get model info
                    cursor.execute("SELECT file_size FROM uploaded_models WHERE id = ?", (model_id,))
                    model_result = cursor.fetchone()
                    model_size = model_result[0] if model_result else 0
                    
                    # Get data info
                    cursor.execute("SELECT data_path FROM training_data WHERE id = ?", (data_id,))
                    data_result = cursor.fetchone()
                    conn.close()
                    
                    interactions_count = 0
                    if data_result and data_result[0] and os.path.exists(data_result[0]):
                        try:
                            with open(data_result[0], 'r') as f:
                                data_content = json.load(f)
                                interactions_count = len(data_content.get('interactions', []))
                        except:
                            pass
                    
                    # Estimate based on both factors
                    estimated_time = estimate_training_time(interactions_count, 1)
                    metadata['interactions_count'] = interactions_count
                    metadata['model_size'] = model_size
            
            except Exception as e:
                logger.error(f"Error estimating training time: {str(e)}")
                # Continue with default estimate
            
            # Store the estimated time in metadata
            metadata['estimated_minutes'] = estimated_time
            
            # Create the training job in the database
            job_id = None
            try:
                with db_lock:
                    job_id = create_training_job(
                        DB_PATH,
                        device_id=device_id,
                        source_type=source_type,
                        source_id=source_id,
                        metadata=metadata
                    )
                
                if not job_id:
                    return jsonify({'success': False, 'message': 'Failed to create training job'}), 500
                
                # Start training in a background thread
                from learning.trainer import trigger_retraining_job
                training_thread = threading.Thread(
                    target=trigger_retraining_job,
                    args=(DB_PATH, job_id, source_type, source_id),
                    daemon=True
                )
                training_thread.start()
                
                # Return success response with job details
                from learning.trainer import get_training_status
                initial_status = get_training_status(DB_PATH, job_id)
                
                response = {
                    'success': True,
                    'message': 'Training job started successfully',
                    'jobId': job_id,
                    'status': 'queued',
                    'sourceType': source_type,
                    'estimatedTimeMinutes': estimated_time,
                    'estimatedCompletionTime': (datetime.now() + timedelta(minutes=estimated_time)).isoformat(),
                    'statusCheckUrl': f"https://{request.host}/api/ai/model-services?operation=status&jobId={job_id}"
                }
                
                # Add job details if available from initial status
                if initial_status and initial_status.get('success') and 'job' in initial_status:
                    response['jobDetails'] = initial_status['job']
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error creating training job: {str(e)}")
                return jsonify({'success': False, 'message': f'Error creating training job: {str(e)}'}), 500
        
        # ======== CHECK TRAINING STATUS OPERATION ========
        elif operation == 'status' and request.method == 'GET':
            from utils.db_helpers import sanitize_string, validate_uuid
            
            job_id = sanitize_string(request.args.get('jobId'), 36)  # UUIDs are 36 chars
            device_id = sanitize_string(request.args.get('deviceId'), 100)
            
            # Validate at least one parameter is provided
            if not job_id and not device_id:
                return jsonify({
                    'success': False, 
                    'message': 'Either jobId or deviceId must be provided',
                    'example': f"https://{request.host}/api/ai/model-services?operation=status&jobId=your-job-id"
                }), 400
            
            # Validate job ID format if provided
            if job_id and not validate_uuid(job_id):
                return jsonify({'success': False, 'message': f'Invalid job ID format: {job_id}'}), 400
            
            # Get specific job status
            if job_id:
                from learning.trainer import get_training_status
                status_response = get_training_status(DB_PATH, job_id)
                
                if not status_response.get('success'):
                    return jsonify(status_response), 404
                
                # Add download URLs to the response for completed jobs
                if 'job' in status_response and status_response['job'].get('status') == 'completed':
                    if status_response['job'].get('resultModelVersion'):
                        status_response['job']['modelDownloadURL'] = (
                            f"https://{request.host}/api/ai/models/{status_response['job']['resultModelVersion']}"
                        )
                
                return jsonify(status_response)
            
            # Get all jobs for a device
            elif device_id:
                # Handle pagination and limiting
                try:
                    limit = int(request.args.get('limit', 10))
                    # Cap limit at reasonable value
                    limit = min(max(1, limit), 50)
                except (ValueError, TypeError):
                    limit = 10
                
                # Get jobs from database
                jobs = get_device_training_jobs(DB_PATH, device_id, limit)
                
                # Add download URLs to completed jobs
                formatted_jobs = []
                for job in jobs:
                    job_data = {
                        'id': job['id'],
                        'status': job['status'],
                        'progress': job['progress'],
                        'sourceType': job.get('source_type'),
                        'createdAt': job['created_at'],
                        'startTime': job['start_time'],
                        'estimatedCompletionTime': job['estimated_completion_time'],
                        'actualCompletionTime': job['actual_completion_time']
                    }
                    
                    # Handle completed jobs with models
                    if job['status'] == 'completed' and job.get('result_model_version'):
                        job_data['resultModelVersion'] = job['result_model_version']
                        job_data['modelDownloadURL'] = f"https://{request.host}/api/ai/models/{job['result_model_version']}"
                        
                        # Add details URL for the specific job
                        job_data['detailsUrl'] = f"https://{request.host}/api/ai/model-services?operation=status&jobId={job['id']}"
                    
                    # Add error message for failed jobs
                    if job['status'] == 'failed' and job.get('error_message'):
                        job_data['errorMessage'] = job['error_message']
                    
                    # Add time remaining for in-progress jobs
                    if job['status'] == 'processing' and job['progress'] > 0:
                        try:
                            # Calculate remaining time if possible
                            if job['start_time'] and job['estimated_completion_time']:
                                now = datetime.now()
                                estimated_completion = datetime.fromisoformat(job['estimated_completion_time'])
                                
                                if estimated_completion > now:
                                    from learning.trainer import format_time_remaining
                                    remaining_seconds = (estimated_completion - now).total_seconds()
                                    job_data['timeRemainingSeconds'] = int(remaining_seconds)
                                    job_data['timeRemainingFormatted'] = format_time_remaining(remaining_seconds)
                        except Exception as e:
                            logger.error(f"Error calculating time remaining: {str(e)}")
                    
                    formatted_jobs.append(job_data)
                
                # Return paginated response with count
                return jsonify({
                    'success': True,
                    'count': len(formatted_jobs),
                    'limit': limit,
                    'deviceId': device_id,
                    'jobs': formatted_jobs
                })
        
        # ======== DOWNLOAD MODEL OPERATION ========
        elif operation == 'download' and request.method == 'GET':
            from utils.db_helpers import sanitize_string
            
            # Get and validate the version parameter
            version = sanitize_string(request.args.get('version'), 50)
            
            if not version:
                # Return the latest model if no version specified
                model_info = get_latest_model_info()
                version = model_info['version']
                
                # Check if we found a valid model
                if not version or not model_info.get('path') or not os.path.exists(model_info.get('path', '')):
                    return jsonify({
                        'success': False, 
                        'message': 'No models available for download',
                        'availableVersions': [v['version'] for v in get_available_model_versions()]
                    }), 404
            
            # Security: Validate version format to prevent path traversal
            if not re.match(r'^[0-9.]+$', version):
                return jsonify({'success': False, 'message': 'Invalid version format'}), 400
            
            # Try to locate the model file
            model_path = os.path.join(MODEL_DIR, f"model_{version}.mlmodel")
            
            # Check if file exists and is readable
            if not os.path.exists(model_path):
                return jsonify({
                    'success': False, 
                    'message': f'Model version {version} not found',
                    'availableVersions': [v['version'] for v in get_available_model_versions()]
                }), 404
            
            if not os.access(model_path, os.R_OK):
                return jsonify({'success': False, 'message': 'Cannot access model file (permission denied)'}), 403
            
            # Check file size to ensure it's not empty or corrupted
            try:
                file_size = os.path.getsize(model_path)
                if file_size == 0:
                    return jsonify({'success': False, 'message': 'Model file is empty'}), 500
            except OSError as e:
                logger.error(f"Error checking model file: {str(e)}")
                return jsonify({'success': False, 'message': 'Error accessing model file'}), 500
            
            # Set security headers for download
            headers = {
                'Content-Disposition': f'attachment; filename="model_{version}.mlmodel"',
                'X-Content-Type-Options': 'nosniff',
                'Content-Type': 'application/octet-stream'
            }
            
            # Log the download attempt
            logger.info(f"Serving model version {version} ({file_size} bytes)")
            
            # Send the file with security headers
            try:
                return send_file(
                    model_path, 
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name=f"model_{version}.mlmodel",
                    etag=True,
                    conditional=True,
                    last_modified=datetime.fromtimestamp(os.path.getmtime(model_path))
                )
            except Exception as e:
                logger.error(f"Error sending model file: {str(e)}")
                return jsonify({'success': False, 'message': f'Error retrieving model: {str(e)}'}), 500
        
        # If we get here, the operation was valid but the method was wrong
        return jsonify({
            'success': False, 
            'message': f'Method {request.method} not allowed for operation {operation}',
            'allowedMethods': {'upload-model': ['POST'], 'upload-data': ['POST'], 'train': ['POST'], 'status': ['GET'], 'download': ['GET']}[operation]
        }), 405
        
    except Exception as e:
        # Log the full exception with traceback for debugging
        logger.error(f"Unhandled error in model-services endpoint: {str(e)}", exc_info=True)
        
        # Return a sanitized error message to the client
        error_message = str(e)
        # Avoid leaking sensitive information in error messages
        if any(sensitive in error_message.lower() for sensitive in ['password', 'secret', 'token', 'key', 'auth']):
            error_message = "An internal server error occurred"
            
        return jsonify({
            'success': False, 
            'message': 'Server error',
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }), 500

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
    # Schedule model training (early morning)
    schedule.every().day.at("02:00").do(train_model_job)
    
    # Schedule cleanup job (late night)
    schedule.every().day.at("03:30").do(cleanup_old_data)
    
    # Schedule disk space check (every 6 hours)
    def check_disk_space():
        try:
            import shutil
            disk_usage = shutil.disk_usage(BASE_DIR)
            percent_free = (disk_usage.free / disk_usage.total) * 100
            if percent_free < 10:
                logger.warning(f"Low disk space: {percent_free:.1f}% free ({disk_usage.free / (1024*1024*1024):.2f} GB)")
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
    
    schedule.every(6).hours.do(check_disk_space)
    
    # Run scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)

# Start scheduler thread
scheduler_thread = threading.Thread(target=run_scheduler)
scheduler_thread.daemon = True
scheduler_thread.start()

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
    # HTML template with API docs and copy feature
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
                margin-bottom: 40px;
                text-align: center;
                padding-bottom: 20px;
                border-bottom: 1px solid var(--border-color);
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
                <h1>Backdoor AI API Documentation</h1>
                <p>API documentation for the Backdoor AI Learning Server</p>
            </header>
            
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