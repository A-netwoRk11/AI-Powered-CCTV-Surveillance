#!/usr/bin/env python3
"""
Video Object Detection Web Interface - Production Ready
Upload video → Get analyzed video with object detection
"""

import os
import sys
import json
import subprocess
import datetime
import shutil
import signal
import webbrowser
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Try to import config, create defaults if fails
try:
    from config.settings import *
    print("✅ Config imported successfully")
except Exception as e:
    print(f"⚠️ Config import failed: {e}")
    print("🔄 Using fallback configuration...")
    # Create default paths - ensure they're Path objects for consistency
    BASE_DIR = Path(__file__).parent.parent
    SRC_DIR = BASE_DIR / "src"
    TEMPLATES_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"
    OUTPUT_DIR = BASE_DIR / "output"
    UPLOADS_DIR = OUTPUT_DIR / "uploads"
    OUTPUT_VIDEOS_DIR = OUTPUT_DIR / "videos"
    SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
    MODELS_DIR = BASE_DIR / "models"
    YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
    COCO_NAMES = BASE_DIR / "data" / "coco.names"
    INPUT_DIR = BASE_DIR / "input"
    DEMO_VIDEOS_DIR = INPUT_DIR / "demo_videos"
    TESTS_DIR = BASE_DIR / "tests"
    print(f"📂 Fallback BASE_DIR: {BASE_DIR}")
    print(f"📂 Fallback OUTPUT_DIR: {OUTPUT_DIR}")

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config['SECRET_KEY'] = 'ai-cctv-surveillance-secret-key-2024'
app.config['UPLOAD_FOLDER'] = str(UPLOADS_DIR)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Reduced to 100MB for server

def create_output_structure():
    """Create comprehensive output folder structure"""
    try:
        # Core output directories
        directories = [
            OUTPUT_DIR,
            OUTPUT_VIDEOS_DIR,
            SCREENSHOTS_DIR,
            UPLOADS_DIR,
            STATIC_DIR / 'saved-test',
            STATIC_DIR / 'results'
        ]
        
        # Create all directories
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created/verified: {directory}")
        
        # Create timestamp-based subdirectories in saved-test (sample structure)
        # This ensures the folder structure is ready for saving analysis results
        try:
            sample_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_dir = STATIC_DIR / 'saved-test' / sample_timestamp
            os.makedirs(sample_dir, exist_ok=True)
            print(f"📁 Created sample timestamp dir: {sample_dir}")
        except Exception as e:
            print(f"⚠️ Could not create sample timestamp dir: {e}")
        
        print(f"✅ Complete output structure created at: {OUTPUT_DIR}")
        return True
    except Exception as e:
        print(f"❌ Failed to create output structure: {e}")
        return False

@app.template_global()
def zip_lists(*args):
    return zip(*args)

app.jinja_env.globals.update(zip=zip)

# Create output structure at startup
create_output_structure()

# Try to load AI models (optional for server startup)
model = None
labels = []

try:
    # Import AI libraries only when needed
    from ultralytics import YOLO
    import cv2
    import numpy as np
    
    if YOLO_MODEL.exists():
        model = YOLO(str(YOLO_MODEL))
    else:
        print(f"⚠️ YOLO model not found at {YOLO_MODEL}")
        print("🔄 Will download YOLO model on first use...")
        model = YOLO('yolov8n.pt')  # This auto-downloads
        print("✅ YOLO model downloaded successfully")
    
    if COCO_NAMES.exists():
        labels = open(str(COCO_NAMES)).read().strip().split("\n")
    else:
        labels = []
        print(f"⚠️ Warning: COCO names not found at {COCO_NAMES}")
        
except Exception as e:
    model = None
    labels = []
    print(f"⚠️ Warning: AI models not loaded: {e}")
    print("🔄 Models will be loaded when needed")

def check_dependencies():
    print("🔍 Checking system dependencies...")
    
    required_dirs = [TEMPLATES_DIR, STATIC_DIR, OUTPUT_DIR]
    missing_dirs = []
    
    for directory in required_dirs:
        if not directory.exists():
            missing_dirs.append(directory)
            print(f"❌ Missing directory: {directory}")
        else:
            print(f"✅ Found directory: {directory}")
    
    # Check optional files (won't fail startup if missing)
    if not YOLO_MODEL.exists():
        print(f"⚠️ YOLO model not found: {YOLO_MODEL} (will auto-download)")
    else:
        print(f"✅ Found YOLO model: {YOLO_MODEL}")
    
    if not COCO_NAMES.exists():
        print(f"⚠️ COCO names not found: {COCO_NAMES} (using defaults)")
    else:
        print(f"✅ Found COCO names: {COCO_NAMES}")
    
    # Only fail if critical directories are missing
    if missing_dirs:
        print(f"❌ Critical directories missing: {missing_dirs}")
        return False
    
    print("✅ All critical dependencies found!")
    return True

def open_browser():
    import time
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

def signal_handler(signum, frame):
    print("\n🛑 Shutting down surveillance system...")
    sys.exit(0)

@app.route('/')
def index():
    """Home page with video upload interface"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Analyze uploaded video using YOLO detection with surveillanceCam.py"""
    print("🔥 ANALYZE REQUEST RECEIVED!")
    print(f"📋 Request method: {request.method}")
    print(f"📋 Request files: {list(request.files.keys())}")
    print(f"📋 Request form: {dict(request.form)}")
    
    try:
        if 'videoFile' not in request.files:
            return render_template('error.html', error='No video file uploaded')
        
        file = request.files['videoFile']
        if file.filename == '':
            return render_template('error.html', error='No video file selected')
        
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        filepath = UPLOADS_DIR / safe_filename
        file.save(str(filepath))
        
        test_name = request.form.get('test_name', 'Analysis')
        prompt = request.form.get('prompt', 'Detect objects and activities')
        
        print(f"🎬 Starting surveillance analysis on uploaded video: {safe_filename}")
        print(f"📁 File saved to: {filepath}")
        print(f"📁 File exists: {filepath.exists()}")
        print(f"📁 File size: {filepath.stat().st_size if filepath.exists() else 'File not found'}")
        
        print("🔄 Importing surveillanceCam module...")
        from surveillanceCam import process_video
        print("✅ Module imported successfully")
        
        print(f"🎥 Starting video processing...")
        print(f"📂 Output directory: {OUTPUT_VIDEOS_DIR}")
        detection_results = process_video(str(filepath), OUTPUT_VIDEOS_DIR)
        print(f"🎯 Detection results: {detection_results}")
        
        if detection_results is None:
            return render_template('error.html', error='Failed to process video file')
        
        detections = []
        for obj_name, count in detection_results.get('detections', {}).items():
            detections.append({
                "object": obj_name,
                "count": count,
                "confidence": 0.8
            })
        
        total_objects = len(detections)
        total_detections = sum(d['count'] for d in detections)
        analysis_summary = f"Detected {total_objects} different object types with {total_detections} total detections"
        
        if detection_results.get('person_detected', 0) > 0:
            analysis_summary += f" including {detection_results['person_detected']} person detections"
        
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
            'screenshot_saved': detection_results.get('screenshot_saved', False)
        }
        
        result_dir = STATIC_DIR / 'saved-test' / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(str(filepath), str(result_dir / safe_filename))
        
        if detection_results.get('output_file') and os.path.exists(detection_results['output_file']):
            analyzed_filename = Path(detection_results['output_file']).name
            shutil.copy2(detection_results['output_file'], str(result_dir / analyzed_filename))
            result_data['analyzed_video'] = analyzed_filename
        
        with open(result_dir / 'metadata.json', 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"✅ Analysis complete! Results saved to: {result_dir}")
        
        return render_template('saved_results.html', 
                             video_filename=safe_filename,
                             detections=detections,
                             test_name=test_name,
                             analysis_summary=analysis_summary,
                             total_frames=result_data['total_frames'],
                             person_detected=result_data['person_detected'],
                             objects_found=result_data['objects_found'],
                             analyzed_video=result_data.get('analyzed_video', ''),
                             screenshot_saved=result_data['screenshot_saved'],
                             is_saved=False)
                             
    except Exception as e:
        import traceback
        error_msg = f"Analysis failed: {str(e)}"
        print(f"❌ Analysis error: {error_msg}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return render_template('error.html', error=error_msg)

@app.route('/saved_analysis')
def saved_analysis():
    try:
        saved_tests = []
        saved_dir = STATIC_DIR / 'saved-test'
        
        if saved_dir.exists():
            for test_dir in saved_dir.iterdir():
                if test_dir.is_dir():
                    metadata_file = test_dir / 'metadata.json'
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            video_files = list(test_dir.glob('*.mp4'))
                            if video_files:
                                metadata['original_video'] = video_files[0].name
                                metadata['dir_name'] = test_dir.name
                                saved_tests.append(metadata)
                        except Exception as e:
                            print(f"Error loading metadata from {metadata_file}: {e}")
        
        saved_tests.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return render_template('saved_analysis.html', tests=saved_tests)
        
    except Exception as e:
        return render_template('error.html', error=f'Error loading saved analyses: {str(e)}')

@app.route('/view_saved/<dir_name>')
def view_saved_test(dir_name):
    """View a specific saved test result"""
    try:
        test_dir = STATIC_DIR / 'saved-test' / dir_name
        metadata_file = test_dir / 'metadata.json'
        
        if not metadata_file.exists():
            return render_template('error.html', error='Test not found')
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        video_files = list(test_dir.glob('*.mp4'))
        if video_files:
            original_video = video_files[0].name
        else:
            original_video = None
        
        return render_template('saved_results.html',
                             metadata=metadata,
                             dir_name=dir_name,
                             original_video=original_video,
                             detections=metadata.get('detections', []))
        
    except Exception as e:
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
            print(f"✅ Analyzed video saved: {analyzed_filename}")
        
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
        print(f"❌ Save analysis error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to save analysis: {str(e)}'
        }), 500

def create_output_structure():
    """Create output directory structure if it doesn't exist"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
        os.makedirs(SCREENSHOTS_DIR, exist_ok=True) 
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        print(f"✅ Output structure created at: {OUTPUT_DIR}")
        return True
    except Exception as e:
        print(f"❌ Failed to create output structure: {e}")
        return False

@app.route('/create_output_folder', methods=['POST'])
def create_output_folder():
    try:
        if create_output_structure():
            return jsonify({
                'status': 'success',
                'message': 'Output folder structure created successfully!',
                'paths': {
                    'output': str(OUTPUT_DIR),
                    'videos': str(OUTPUT_VIDEOS_DIR),
                    'screenshots': str(SCREENSHOTS_DIR),
                    'uploads': str(UPLOADS_DIR)
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to create output folder structure'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error creating folders: {str(e)}'
        }), 500

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
        
        print(f"🎥 Starting live webcam recording for {duration} seconds...")
        
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
        
        print(f"🔍 Analyzing live recording: {latest_recording.name}")
        
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

if __name__ == '__main__':
    print("🚀 Starting AI-Powered CCTV Surveillance Web Interface...")
    print(f"📂 Base Directory: {BASE_DIR}")
    print(f"📂 Templates: {TEMPLATES_DIR}")
    print(f"📂 Static: {STATIC_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}")
    print(f"📂 Uploads: {UPLOADS_DIR}")
    print(f"🤖 YOLO Model: {'✅ Loaded' if model else '❌ Failed'}")
    print(f"🏷️  Labels: {len(labels)} classes loaded")
    
    # Ensure output structure exists
    print("🔧 Ensuring output structure exists...")
    create_output_structure()
    
    print("🌐 Server starting...")
    print(f"🌍 Host: 0.0.0.0")
    print(f"🔌 Port: {int(os.environ.get('PORT', 5000))}")
    
    if check_dependencies():
        print("✅ All dependencies checked")
        # Don't open browser on server deployment
        pass
    else:
        print("⚠️ Some dependencies missing, but continuing...")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
