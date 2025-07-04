#!/usr/bin/env python3
"""
AI-Powered CCTV Surveillance Web Interface

A Flask-based web application for video surveillance analysis using YOLO object detection.
Upload videos → Get analyzed videos with object detection and alerts.
"""

import os
import sys
import json
import subprocess
import datetime
import shutil
import logging
import threading
import signal
import webbrowser
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['LEVEL']),
    format=LOGGING_CONFIG['FORMAT'],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGGING_CONFIG['LOG_FILE']) if LOGGING_CONFIG['LOG_FILE'].parent.exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app with proper configuration
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))

# Apply configuration from settings
app.config.update(WEB_CONFIG)
app.config.update(SECURITY_CONFIG)
app.config['UPLOAD_FOLDER'] = str(UPLOADS_DIR)

# Global variables for model and labels
model = None
labels = []

def initialize_model():
    """Initialize YOLO model and labels with proper error handling."""
    global model, labels
    
    try:
        if YOLO_MODEL.exists():
            model = YOLO(str(YOLO_MODEL))
            logger.info(f"YOLO model loaded successfully: {YOLO_MODEL}")
        else:
            logger.warning(f"YOLO model not found at {YOLO_MODEL}")
            logger.info("Model will be downloaded automatically on first use")
            model = YOLO('yolov8n.pt')  # This will download the model
        
        if COCO_NAMES.exists():
            labels = open(str(COCO_NAMES)).read().strip().split("\n")
            logger.info(f"Loaded {len(labels)} COCO class labels")
        else:
            logger.warning(f"COCO names not found at {COCO_NAMES}")
            # Use default COCO labels if file is missing
            labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
            
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        model = None
        labels = []

def create_output_structure():
    """Create output directory structure with proper error handling."""
    try:
        # Create all necessary directories
        directories = [
            OUTPUT_DIR, OUTPUT_VIDEOS_DIR, SCREENSHOTS_DIR, UPLOADS_DIR,
            RESULTS_DIR, SAVED_ANALYSIS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        test_file = OUTPUT_DIR / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        logger.info(f"Output structure created at: {OUTPUT_DIR.absolute()}")
        return True
        
    except PermissionError as e:
        logger.error(f"Permission denied creating output structure: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to create output structure: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies and files are present."""
    logger.info("Checking system dependencies...")
    
    required_dirs = [MODELS_DIR, TEMPLATES_DIR, STATIC_DIR, OUTPUT_DIR]
    missing_dirs = [d for d in required_dirs if not d.exists()]
    
    if missing_dirs:
        for directory in missing_dirs:
            logger.error(f"Missing directory: {directory}")
        return False
    
    if not COCO_NAMES.exists():
        logger.warning(f"Missing COCO names: {COCO_NAMES}")
        # This is not critical as we have fallback labels
    
    logger.info("All dependencies found!")
    return True

# Initialize the application
if not validate_configuration():
    logger.error("[ERROR] Configuration validation failed")
    sys.exit(1)

create_output_structure()
initialize_model()

# Template utilities
@app.template_global()
def zip_lists(*args):
    return zip(*args)

app.jinja_env.globals.update(zip=zip)

@app.route('/')
def index():
    """Home page with video upload interface"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video using YOLO detection with surveillanceCam.py"""
    logger.info("[INFO] ANALYZE REQUEST RECEIVED!")
    logger.debug(f"Request method: {request.method}")
    logger.debug(f"Request files: {list(request.files.keys())}")
    logger.debug(f"Request form: {dict(request.form)}")
    
    try:
        # Validate file upload
        if 'videoFile' not in request.files:
            logger.warning("No video file in request")
            return render_template('error.html', error='No video file uploaded')
        
        file = request.files['videoFile']
        if file.filename == '':
            logger.warning("Empty filename")
            return render_template('error.html', error='No video file selected')
        
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in WEB_CONFIG['UPLOAD_EXTENSIONS']:
            logger.warning(f"Invalid file extension: {file_ext}")
            return render_template('error.html', 
                                 error=f'Invalid file type. Allowed: {", ".join(WEB_CONFIG["UPLOAD_EXTENSIONS"])}')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        filepath = UPLOADS_DIR / safe_filename
        file.save(str(filepath))
        
        # Get form data
        test_name = request.form.get('test_name', 'Analysis')
        prompt = request.form.get('prompt', 'Detect objects and activities')
        
        logger.info(f"[INFO] Starting surveillance analysis on: {safe_filename}")
        logger.debug(f"File saved to: {filepath}")
        logger.debug(f"File size: {filepath.stat().st_size} bytes")
        
        # Import and run analysis
        try:
            from surveillanceCam import process_video
            logger.debug("[OK] Module imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import surveillanceCam: {e}")
            return render_template('error.html', error='Analysis module not available')
        
        logger.info("[INFO] Starting video processing...")
        detection_results = process_video(str(filepath), OUTPUT_VIDEOS_DIR)
        
        if detection_results is None:
            logger.error("Video processing returned None")
            return render_template('error.html', error='Failed to process video file')
        
        # Process detection results
        detections = []
        for obj_name, count in detection_results.get('detections', {}).items():
            detections.append({
                "object": obj_name,
                "count": count,
                "confidence": 0.8  # Average confidence placeholder
            })
        
        # Generate analysis summary
        total_objects = len(detections)
        total_detections = sum(d['count'] for d in detections)
        analysis_summary = f"Detected {total_objects} different object types with {total_detections} total detections"
        
        if detection_results.get('person_detected', 0) > 0:
            analysis_summary += f" including {detection_results['person_detected']} person detections"
        
        # Prepare result data
        result_data = {
            'test_name': test_name,
            'prompt': prompt,
            'video_filename': safe_filename,
            'detections': detections,
            'timestamp': timestamp,
            'analysis_summary': analysis_summary,
            'total_frames': detection_results.get('total_frames', 0),
            'person_detected': detection_results.get('person_detected', 0),
            'objects_found': detection_results.get('objects_found', []),
            'output_video': detection_results.get('output_file', ''),
            'screenshot_saved': detection_results.get('screenshot_saved', False),
            'processing_time': detection_results.get('processing_time', 0)
        }
        
        # Save results
        result_dir = SAVED_ANALYSIS_DIR / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original video
        shutil.copy2(str(filepath), str(result_dir / safe_filename))
        
        # Copy analyzed video if available
        if detection_results.get('output_file') and os.path.exists(detection_results['output_file']):
            analyzed_filename = Path(detection_results['output_file']).name
            shutil.copy2(detection_results['output_file'], str(result_dir / analyzed_filename))
            result_data['analyzed_video'] = analyzed_filename
        
        # Save metadata
        with open(result_dir / 'metadata.json', 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"[OK] Analysis complete! Results saved to: {result_dir}")
        
        # Use the existing saved_results.html template
        # Create metadata object that matches template expectations
        metadata_for_template = {
            'test_name': test_name,
            'timestamp': timestamp,
            'total_frames': result_data['total_frames'],
            'person_detected': result_data['person_detected'],
            'analysis_summary': analysis_summary,
            'analyzed_video': result_data.get('analyzed_video', '')
        }
        
        return render_template('saved_results.html', 
                             metadata=metadata_for_template,
                             detections=detections,
                             analyzed_video=result_data.get('analyzed_video', ''),
                             is_saved=False)
                             
    except Exception as e:
        import traceback
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(f"[ERROR] Analysis error: {error_msg}")
        logger.debug("Full traceback:", exc_info=True)
        return render_template('error.html', error=error_msg)

@app.route('/saved_analysis')
def saved_analysis():
    """Display all saved analysis results."""
    try:
        saved_tests = []
        
        if SAVED_ANALYSIS_DIR.exists():
            for test_dir in SAVED_ANALYSIS_DIR.iterdir():
                if test_dir.is_dir():
                    metadata_file = test_dir / 'metadata.json'
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Find video files in the directory
                            video_files = list(test_dir.glob('*.mp4'))
                            if video_files:
                                metadata['original_video'] = video_files[0].name
                                metadata['dir_name'] = test_dir.name
                                saved_tests.append(metadata)
                        except Exception as e:
                            logger.error(f"Error loading metadata from {metadata_file}: {e}")
        
        # Sort by timestamp (newest first)
        saved_tests.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        logger.info(f"Found {len(saved_tests)} saved analyses")
        
        return render_template('saved_analysis.html', tests=saved_tests)
        
    except Exception as e:
        logger.error(f"Error loading saved analyses: {e}")
        return render_template('error.html', error=f'Error loading saved analyses: {str(e)}')

@app.route('/view_saved/<dir_name>')
def view_saved_test(dir_name):
    """View a specific saved test result."""
    try:
        test_dir = SAVED_ANALYSIS_DIR / dir_name
        metadata_file = test_dir / 'metadata.json'
        
        if not metadata_file.exists():
            logger.warning(f"Metadata not found for test: {dir_name}")
            return render_template('error.html', error='Test not found')
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Find video files
        video_files = list(test_dir.glob('*.mp4'))
        original_video = video_files[0].name if video_files else None
        
        logger.info(f"Viewing saved test: {dir_name}")
        
        # Use existing saved_results.html template
        return render_template('saved_results.html',
                             metadata=metadata,
                             dir_name=dir_name,
                             original_video=original_video,
                             detections=metadata.get('detections', []),
                             test_name=metadata.get('test_name', 'Unknown'),
                             analysis_summary=metadata.get('analysis_summary', ''),
                             total_frames=metadata.get('total_frames', 0),
                             person_detected=metadata.get('person_detected', 0),
                             objects_found=metadata.get('objects_found', []),
                             analyzed_video=metadata.get('analyzed_video', ''),
                             screenshot_saved=metadata.get('screenshot_saved', False),
                             processing_time=metadata.get('processing_time', 0),
                             timestamp=metadata.get('timestamp', ''),
                             is_saved=True)
        
    except Exception as e:
        logger.error(f"Error viewing test {dir_name}: {e}")
        return render_template('error.html', error=f'Error viewing test: {str(e)}')

@app.route('/test-upload', methods=['GET', 'POST'])
def test_upload():
    """Simple test upload page"""
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('test_upload.html', error='No file uploaded')
            
            file = request.files['file']
            if file.filename == '':
                return render_template('test_upload.html', error='No file selected')
            
            filename = secure_filename(file.filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"test_{timestamp}_{filename}"
            filepath = UPLOADS_DIR / safe_filename
            file.save(str(filepath))
            
            return render_template('test_upload.html', 
                                 success=f'File uploaded successfully: {safe_filename}',
                                 video_path=f'/uploads/{safe_filename}')
        except Exception as e:
            return render_template('test_upload.html', error=f'Upload failed: {str(e)}')
    
    return render_template('test_upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(UPLOADS_DIR / filename)

@app.route('/analyzed_video/<filename>')
def analyzed_video(filename):
    try:
        video_path = OUTPUT_VIDEOS_DIR / filename
        if video_path.exists():
            return send_file(str(video_path))
        else:
            return "Video not found", 404
    except Exception as e:
        return f"Error serving video: {str(e)}", 500

@app.route('/video/<filename>')
def serve_video(filename):
    video_path = OUTPUT_VIDEOS_DIR / filename
    if video_path.exists():
        return send_file(str(video_path))
    else:
        return jsonify({'error': 'Video not found'}), 404

@app.route('/upload/<filename>')
def serve_upload(filename):
    video_path = UPLOADS_DIR / filename
    if video_path.exists():
        return send_file(str(video_path))
    else:
        return jsonify({'error': 'Video not found'}), 404

@app.route('/run_webcam')
def run_webcam():
    try:
        webcam_script = SRC_DIR / 'WebCam.py'
        subprocess.Popen([sys.executable, str(webcam_script)], cwd=str(SRC_DIR))
        return jsonify({'status': 'success', 'message': 'Webcam surveillance started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/run_surveillance')
def run_surveillance():
    try:
        surveillance_script = SRC_DIR / 'surveillanceCam.py'
        subprocess.Popen([sys.executable, str(surveillance_script)], cwd=str(SRC_DIR))
        return jsonify({'status': 'success', 'message': 'Video surveillance started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/send_email')
def send_email():
    try:
        email_script = SRC_DIR / 'sendGmail.py'
        subprocess.run([sys.executable, str(email_script)], cwd=str(SRC_DIR))
        return jsonify({'status': 'success', 'message': 'Email sent successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/debug/info')
def debug_info():
    """Debug information endpoint"""
    info = {
        'base_dir': str(BASE_DIR),
        'src_dir': str(SRC_DIR),
        'templates_dir': str(TEMPLATES_DIR),
        'static_dir': str(STATIC_DIR),
        'uploads_dir': str(UPLOADS_DIR),
        'models_dir': str(MODELS_DIR),
        'yolo_model_exists': YOLO_MODEL.exists(),
        'coco_names_exists': COCO_NAMES.exists(),
        'model_loaded': model is not None,
        'labels_count': len(labels)
    }
    return jsonify(info)

@app.route('/save_analysis', methods=['POST'])
def save_analysis():
    try:
        data = request.get_json()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dir = STATIC_DIR / 'saved-test' / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        
        latest_video = None
        if OUTPUT_VIDEOS_DIR.exists():
            video_files = list(OUTPUT_VIDEOS_DIR.glob("*.mp4"))
            if video_files:
                latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
        
        if latest_video and latest_video.exists():
            analyzed_filename = f"analyzed_{timestamp}.mp4"
            shutil.copy2(str(latest_video), str(save_dir / analyzed_filename))
            print(f"[OK] Analyzed video saved: {analyzed_filename}")
        
        metadata = {
            'timestamp': timestamp,
            'test_name': data.get('test_name', 'Analysis'),
            'analyzed_video': analyzed_filename if latest_video else None,
            'original_video': data.get('original_video', ''),
            'titles': data.get('titles', []),
            'time_frames': data.get('time_frames', []),
            'metadata': data.get('metadata', {})
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': 'Analysis saved successfully with analyzed video!',
            'saved_dir': str(save_dir)
        })
        
    except Exception as e:
        print(f"[ERROR] Save analysis error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to save analysis: {str(e)}'
        }), 500

def create_output_structure():
    """Create output directory structure if it doesn't exist - kept for API compatibility."""
    try:
        directories = [OUTPUT_DIR, OUTPUT_VIDEOS_DIR, SCREENSHOTS_DIR, UPLOADS_DIR]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[OK] Output structure created at: {OUTPUT_DIR}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to create output structure: {e}")
        return False

@app.route('/create_output_folder', methods=['POST'])
def create_output_folder():
    """API endpoint to create output folder structure."""
    try:
        # Get absolute paths for response
        paths = {
            'output': str(OUTPUT_DIR.absolute()),
            'videos': str(OUTPUT_VIDEOS_DIR.absolute()),
            'screenshots': str(SCREENSHOTS_DIR.absolute()),
            'uploads': str(UPLOADS_DIR.absolute())
        }
        
        if create_output_structure():
            return jsonify({
                'status': 'success',
                'message': 'Output folder structure created successfully!',
                'paths': paths
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to create output folder structure. Check file permissions and disk space.'
            }), 500
            
    except PermissionError as e:
        return jsonify({
            'status': 'error',
            'message': f'Permission denied: Cannot create folders in this location. {str(e)}'
        }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error creating folders: {str(e)}'
        }), 500

def open_browser():
    """Open the web browser after a delay."""
    import time
    time.sleep(2)
    webbrowser.open(f'http://{WEB_CONFIG["HOST"]}:{WEB_CONFIG["PORT"]}')

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("\n[INFO] Shutting down surveillance system...")
    sys.exit(0)

@app.route('/open_video_folder')
def open_video_folder():
    try:
        import subprocess
        import os
        
        video_folder = str(OUTPUT_VIDEOS_DIR)
        
        # Ensure folder exists
        os.makedirs(video_folder, exist_ok=True)
        
        # Open folder in Windows Explorer (don't use check=True to avoid exceptions)
        subprocess.run(['explorer', video_folder])
        
        return jsonify({
            'status': 'success',
            'message': 'Video folder opened successfully!',
            'folder_path': video_folder
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to open video folder: {str(e)}'
        }), 500

@app.route('/record_webcam', methods=['POST'])
def record_webcam():
    try:
        data = request.get_json() if request.is_json else {}
        duration = data.get('duration', 30) #default 30 sec
        
        print(f"[INFO] Starting live webcam recording for {duration} seconds...")
        
        import subprocess
        import threading
        
        def run_webcam_recording():
            webcam_script = SRC_DIR / 'WebCam.py'
            env = os.environ.copy()
            env['RECORDING_DURATION'] = str(duration)
            subprocess.run([sys.executable, str(webcam_script)], 
                         cwd=str(SRC_DIR), env=env)
        
        recording_thread = threading.Thread(target=run_webcam_recording)
        recording_thread.daemon = True
        recording_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Live webcam recording started for {duration} seconds. Video will be saved in uploads folder.',
            'duration': duration
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start webcam recording: {str(e)}'
        }), 500

@app.route('/analyze_live_recording', methods=['POST'])
def analyze_live_recording():
    try:
        live_recordings = []
        for file_path in UPLOADS_DIR.glob("live_recording_*.mp4"):
            live_recordings.append(file_path)
        
        if not live_recordings:
            return jsonify({
                'status': 'error',
                'message': 'No live recordings found to analyze'
            }), 404
        
        latest_recording = max(live_recordings, key=os.path.getctime)
        
        data = request.get_json() if request.is_json else {}
        test_name = data.get('test_name', f'Live_Recording_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
        prompt = data.get('prompt', 'Analyze objects in this live recording')
        
        print(f"[INFO] Analyzing live recording: {latest_recording.name}")
        
        surveillance_script = SRC_DIR / 'surveillanceCam.py'
        result = subprocess.run([
            sys.executable, str(surveillance_script), 
            str(latest_recording), test_name, prompt
        ], capture_output=True, text=True, cwd=str(SRC_DIR))
        
        if result.returncode == 0:
            return jsonify({
                'status': 'success',
                'message': f'Live recording analyzed successfully as "{test_name}"',
                'test_name': test_name,
                'original_file': latest_recording.name
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Analysis failed: {result.stderr}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to analyze live recording: {str(e)}'
        }), 500

@app.route('/delete_analysis/<dir_name>', methods=['DELETE'])
def delete_analysis(dir_name):
    """Delete a saved analysis and its associated files."""
    try:
        # Validate directory name to prevent path traversal
        if '..' in dir_name or '/' in dir_name or '\\' in dir_name:
            return jsonify({
                'status': 'error',
                'message': 'Invalid directory name'
            }), 400
        
        # Construct the full path to the analysis directory
        analysis_dir = SAVED_ANALYSIS_DIR / dir_name
        
        if not analysis_dir.exists():
            return jsonify({
                'status': 'error',
                'message': 'Analysis not found'
            }), 404
        
        # Remove the entire analysis directory and its contents
        import shutil
        shutil.rmtree(str(analysis_dir))
        
        logger.info(f"Analysis deleted: {dir_name}")
        
        return jsonify({
            'status': 'success',
            'message': 'Analysis deleted successfully'
        })
        
    except PermissionError as e:
        logger.error(f"Permission error deleting analysis {dir_name}: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Permission denied: Cannot delete analysis files'
        }), 500
    except Exception as e:
        logger.error(f"Error deleting analysis {dir_name}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to delete analysis: {str(e)}'
        }), 500

if __name__ == '__main__':
    """Main application entry point."""
    logger.info("Starting AI-Powered CCTV Surveillance Web Interface...")
    logger.info(f"Templates: {TEMPLATES_DIR}")
    logger.info(f"Static: {STATIC_DIR}")
    logger.info(f"Uploads: {UPLOADS_DIR}")
    logger.info(f"YOLO Model: {'Loaded' if model else 'Failed'}")
    logger.info(f"Labels: {len(labels)} classes loaded")
    
    # Get configuration summary
    config_summary = get_config_summary()
    logger.info(f"Server starting at http://localhost:{config_summary['web_port']}")
    logger.info(f"Environment: {config_summary['environment']}")
    logger.info(f"Max file size: {config_summary['max_file_size']}")
    
    # Check dependencies and start browser if everything is OK
    if check_dependencies():
        if not DEBUG:  # Only auto-open browser in production
            threading.Thread(target=open_browser, daemon=True).start()
    else:
        logger.warning("Some dependencies are missing, but starting anyway...")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the Flask application
    try:
        # Get port from environment variable (Render provides this)
        port = int(os.environ.get('PORT', WEB_CONFIG['PORT']))
        
        # Use configuration from settings
        app.run(
            debug=WEB_CONFIG['DEBUG'],
            host='0.0.0.0',  # Bind to all interfaces for Render
            port=port,
            threaded=True  # Enable threading for better performance
        )
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        sys.exit(1)

# WSGI entry point for Render deployment
if __name__ != '__main__':
    # When running under Gunicorn, we need to expose the app
    application = app
