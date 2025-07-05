#!/usr/bin/env python3
"""
Video Object Detection Web Interface
Upload video → Get analyzed video with object detection
"""

import os
import sys
import json
import subprocess
import datetime
import shutil
import zipfile
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import webbrowser
import signal
import threading

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = str(UPLOADS_DIR)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

@app.template_global()
def zip_lists(*args):
    return zip(*args)

app.jinja_env.globals.update(zip=zip)

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR / 'saved-test', exist_ok=True)
os.makedirs(STATIC_DIR / 'results', exist_ok=True)

# Initialize model with better error handling for deployment
model = None
labels = []

try:
    # Try local model first, then download
    if YOLO_MODEL.exists():
        print(f"📦 Loading local YOLO model from {YOLO_MODEL}")
        model = YOLO(str(YOLO_MODEL))
        print("✅ Local YOLO model loaded successfully")
    else:
        print("📦 Local model not found, downloading YOLOv8 nano...")
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("✅ Downloaded YOLO model loaded successfully")
    
    # Load COCO names with fallback
    if COCO_NAMES.exists():
        labels = open(str(COCO_NAMES)).read().strip().split("\n")
        print(f"✅ Loaded {len(labels)} COCO class labels from file")
    else:
        print("⚠️ COCO names file not found, using default labels")
        # Default COCO labels as fallback
        labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        print(f"✅ Using default {len(labels)} COCO labels")
        
except Exception as e:
    model = None
    labels = []
    print(f"⚠️ Warning: YOLO model loading failed: {e}")
    print("   Model will be initialized on first request")

def check_dependencies():
    print("🔍 Checking system dependencies...")
    
    required_dirs = [MODELS_DIR, TEMPLATES_DIR, STATIC_DIR, OUTPUT_DIR]
    for directory in required_dirs:
        if not directory.exists():
            print(f"❌ Missing directory: {directory}")
            return False
    
    if not YOLO_MODEL.exists():
        print(f"❌ Missing YOLO model: {YOLO_MODEL}")
        return False
    
    if not COCO_NAMES.exists():
        print(f"❌ Missing COCO names: {COCO_NAMES}")
        return False
    
    print("✅ All dependencies found!")
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
    global model  # Declare global at the beginning
    
    print("🔥 ANALYZE REQUEST RECEIVED!")
    print(f"📋 Request method: {request.method}")
    print(f"📋 Request files: {list(request.files.keys())}")
    print(f"📋 Request form: {dict(request.form)}")
    
    try:
        # Ensure model is loaded for every request (important for Render)
        if model is None:
            print("⚠️ Model not loaded, trying to initialize...")
            try:
                model = YOLO('yolov8n.pt')  # Download if needed
                print("✅ Model initialized successfully")
            except Exception as e:
                print(f"❌ Model initialization failed: {e}")
                return render_template('error.html', error='AI model failed to load. Please try again.')
        
        if 'videoFile' not in request.files:
            return render_template('error.html', error='No video file uploaded')
        
        file = request.files['videoFile']
        if file.filename == '':
            return render_template('error.html', error='No video file selected')
        
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        
        # Create uploads directory if it doesn't exist
        try:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = UPLOADS_DIR / safe_filename
        except Exception as e:
            print(f"⚠️ Could not create uploads dir: {e}")
            # Use temp directory as fallback
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            filepath = temp_dir / safe_filename
            print(f"Using temp directory: {filepath}")
        
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
        
        # Ensure output directory exists
        try:
            OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
            output_dir = OUTPUT_VIDEOS_DIR
        except Exception as e:
            print(f"⚠️ Could not create output dir: {e}")
            import tempfile
            output_dir = Path(tempfile.gettempdir())
            print(f"Using temp output directory: {output_dir}")
        
        print(f"📂 Output directory: {output_dir}")
        detection_results = process_video(str(filepath), output_dir)
        print(f"🎯 Detection results: {detection_results}")
        
        if detection_results is None:
            return render_template('error.html', error='Failed to process video file')
        
        # Process detection results
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
        
        # Save results with error handling
        try:
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
        
        except Exception as save_error:
            print(f"⚠️ Could not save results: {save_error}")
            # Continue without failing - just don't save to disk
        
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
        
        # Ensure directory exists
        if not saved_dir.exists():
            try:
                saved_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created saved analysis directory: {saved_dir}")
            except Exception as e:
                print(f"Could not create saved analysis directory: {e}")
                # Return empty results instead of failing
                return render_template('saved_analysis.html', tests=[])
        
        # Load saved analyses with error handling
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
        print(f"Found {len(saved_tests)} saved analyses")
        return render_template('saved_analysis.html', tests=saved_tests)
        
    except Exception as e:
        print(f"Error in saved_analysis route: {e}")
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

@app.route('/download_analysis/<dir_name>')
def download_analysis(dir_name):
    """Download analyzed video and related files as a ZIP archive"""
    try:
        # Find the analysis directory
        analysis_dir = STATIC_DIR / 'saved-test' / dir_name
        
        if not analysis_dir.exists():
            return jsonify({
                'status': 'error',
                'message': f'Analysis directory not found: {dir_name}'
            }), 404
        
        # Create a temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.close()
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from the analysis directory
            for file_path in analysis_dir.rglob('*'):
                if file_path.is_file():
                    # Add file to ZIP with relative path
                    arc_name = file_path.relative_to(analysis_dir)
                    zipf.write(file_path, arc_name)
            
            # Also look for the analyzed video in the output/videos directory
            video_files = list(OUTPUT_VIDEOS_DIR.glob(f"{dir_name}_analyzed_*.mp4"))
            for video_file in video_files:
                if video_file.exists():
                    zipf.write(video_file, f"analyzed_video_{video_file.name}")
        
        # Generate download filename
        download_name = f"{dir_name}_analysis.zip"
        
        def cleanup_temp_file():
            """Clean up temporary file after download"""
            try:
                os.unlink(temp_zip.name)
            except:
                pass
        
        # Schedule cleanup after response is sent
        threading.Timer(10.0, cleanup_temp_file).start()
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name=download_name,
            mimetype='application/zip'
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to create download archive: {str(e)}'
        }), 500

@app.route('/delete_analysis/<dir_name>', methods=['DELETE'])
def delete_analysis(dir_name):
    """Delete an analysis and all its associated files"""
    try:
        # Find and delete the analysis directory
        analysis_dir = STATIC_DIR / 'saved-test' / dir_name
        
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
            print(f"🗑️ Deleted analysis directory: {analysis_dir}")
        
        # Also delete the analyzed video files
        video_files = list(OUTPUT_VIDEOS_DIR.glob(f"{dir_name}_analyzed_*.mp4"))
        for video_file in video_files:
            if video_file.exists():
                video_file.unlink()
                print(f"🗑️ Deleted video file: {video_file}")
        
        # Delete screenshots
        screenshot_files = list(SCREENSHOTS_DIR.glob(f"*_{dir_name}_*.jpg"))
        for screenshot_file in screenshot_files:
            if screenshot_file.exists():
                screenshot_file.unlink()
                print(f"🗑️ Deleted screenshot: {screenshot_file}")
        
        # Delete upload file
        upload_files = list(UPLOADS_DIR.glob(f"{dir_name}.*"))
        for upload_file in upload_files:
            if upload_file.exists():
                upload_file.unlink()
                print(f"🗑️ Deleted upload file: {upload_file}")
        
        return jsonify({
            'status': 'success',
            'message': f'Analysis "{dir_name}" deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to delete analysis: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 Starting AI-Powered CCTV Surveillance Web Interface...")
    print(f"📂 Templates: {TEMPLATES_DIR}")
    print(f"📂 Static: {STATIC_DIR}")
    print(f"📂 Uploads: {UPLOADS_DIR}")
    print(f"🤖 YOLO Model: {'✅ Loaded' if model else '❌ Failed'}")
    print(f"🏷️  Labels: {len(labels)} classes loaded")
    print("🌐 Server starting at http://localhost:5000")
    
    if check_dependencies():
        threading.Thread(target=open_browser).start()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 5000))
    
    # Force production mode if running on Render
    is_render = os.environ.get('RENDER') or os.environ.get('RENDER_SERVICE_NAME')
    debug_mode = not is_render
    
    if is_render:
        print("🌐 Running on Render - production mode")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)

# WSGI entry point for Render deployment
application = app

# Initialize for deployment
print("🚀 DEPLOYMENT INITIALIZATION:")
print(f"📂 BASE_DIR: {BASE_DIR}")
print(f"📂 Current working directory: {os.getcwd()}")

# Ensure model is loaded for deployment
if model is None:
    print("🤖 Initializing model for deployment...")
    try:
        model = YOLO('yolov8n.pt')  # Will download if needed
        print("✅ Deployment model initialization successful")
    except Exception as e:
        print(f"⚠️ Deployment model initialization failed: {e}")

print("✅ Deployment initialization complete")
