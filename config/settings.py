"""
Configuration settings for AI-Powered CCTV Surveillance System.

This module contains all configuration parameters for the surveillance system,
including paths, model settings, web server configuration, and environment-specific settings.
"""

import os
import sys
from pathlib import Path

# Environment detection
ENV = os.environ.get('FLASK_ENV', os.environ.get('ENVIRONMENT', 'production'))
DEBUG = ENV == 'development'

# Base paths
BASE_DIR = Path(__file__).parent.parent
# Fix for when running from src directory (like with cd src && gunicorn)
# Check if we're running from src directory by checking current working directory
if Path.cwd().name == 'src':
    # We're in src directory, so BASE_DIR calculation is wrong, fix it
    BASE_DIR = Path.cwd().parent
SRC_DIR = BASE_DIR / "src"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
CONFIG_DIR = BASE_DIR / "config"

# Model paths and configurations
MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
COCO_NAMES = BASE_DIR / "data" / "coco.names"

# Alternative model options (can be switched based on requirements)
YOLO_MODELS = {
    'nano': MODELS_DIR / "yolov8n.pt",      # Fastest, lowest accuracy
    'small': MODELS_DIR / "yolov8s.pt",     # Balanced
    'medium': MODELS_DIR / "yolov8m.pt",    # Better accuracy
    'large': MODELS_DIR / "yolov8l.pt",     # Best accuracy, slowest
    'v3': MODELS_DIR / "yolov3.weights"     # Legacy YOLOv3
}

# Detection parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.4
DEFAULT_IMAGE_SIZE = 640
MAX_DETECTION_CLASSES = 80

# Input/Output paths
INPUT_DIR = BASE_DIR / "input"
DEMO_VIDEOS_DIR = INPUT_DIR / "demo_videos"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_VIDEOS_DIR = OUTPUT_DIR / "videos"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
SAVED_ANALYSIS_DIR = STATIC_DIR / "saved-test"

# Test paths
TESTS_DIR = BASE_DIR / "tests"
TEST_OUTPUT_DIR = BASE_DIR / "test_output"

# Web server configuration
WEB_CONFIG = {
    'HOST': os.environ.get('FLASK_HOST', '0.0.0.0'),
    'PORT': int(os.environ.get('PORT', os.environ.get('FLASK_PORT', '5000'))),
    'DEBUG': DEBUG,
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production'),
    'MAX_CONTENT_LENGTH': 500 * 1024 * 1024,  # 500MB max file size
    'UPLOAD_EXTENSIONS': ['.mp4', '.avi', '.mov', '.mkv', '.wmv'],
    'ALLOWED_EXTENSIONS': {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
}

# Performance settings
PERFORMANCE_CONFIG = {
    'DEFAULT_SKIP_FRAMES': 2,           # Process every Nth frame
    'MAX_PROCESSING_TIME': 3600,       # 1 hour max processing time
    'FRAME_BATCH_SIZE': 10,             # Process frames in batches
    'MAX_VIDEO_DURATION': 1800,        # 30 minutes max video length
    'PROGRESS_UPDATE_INTERVAL': 50,     # Update progress every N frames
}

# Recording settings
RECORDING_CONFIG = {
    'DEFAULT_DURATION': 30,             # Default recording duration in seconds
    'MAX_DURATION': 300,                # Maximum recording duration (5 minutes)
    'DEFAULT_FPS': 20,                  # Default frames per second
    'DEFAULT_RESOLUTION': (1080, 720),  # Default resolution (width, height)
    'VIDEO_CODEC': 'mp4v',              # Default video codec
}

# Logging configuration - Use console logging for deployment safety
LOGGING_CONFIG = {
    'LEVEL': 'DEBUG' if DEBUG else 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'LOG_FILE': None,  # Disable file logging for deployment safety
    'MAX_LOG_SIZE': 10 * 1024 * 1024,   # 10MB
    'BACKUP_COUNT': 5
}

# Security settings
SECURITY_CONFIG = {
    'CSRF_ENABLED': True,
    'WTF_CSRF_TIME_LIMIT': None,
    'SESSION_COOKIE_SECURE': not DEBUG,
    'SESSION_COOKIE_HTTPONLY': True,
    'PERMANENT_SESSION_LIFETIME': 1800,  # 30 minutes
}

# Database configuration (for future use)
DATABASE_CONFIG = {
    'DATABASE_URL': os.environ.get('DATABASE_URL', f'sqlite:///{BASE_DIR}/surveillance.db'),
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'SQLALCHEMY_ECHO': DEBUG
}

# Render deployment configuration
RENDER_CONFIG = {
    'PORT': int(os.environ.get('PORT', '5000')),
    'WORKERS': int(os.environ.get('WEB_CONCURRENCY', 1)),
    'TIMEOUT': int(os.environ.get('TIMEOUT', 120)),
}

def validate_configuration():
    """
    Validate configuration and create necessary directories.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        # Create necessary directories
        directories_to_create = [
            OUTPUT_DIR, OUTPUT_VIDEOS_DIR, SCREENSHOTS_DIR, UPLOADS_DIR,
            RESULTS_DIR, SAVED_ANALYSIS_DIR, TEST_OUTPUT_DIR,
            LOGGING_CONFIG['LOG_FILE'].parent
        ]
        
        for directory in directories_to_create:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate critical files exist
        critical_files = [COCO_NAMES]
        
        for file_path in critical_files:
            if not file_path.exists():
                print(f"⚠️ Warning: Critical file missing: {file_path}")
        
        # Check YOLO model availability
        if not YOLO_MODEL.exists():
            print(f"⚠️ Warning: Default YOLO model not found: {YOLO_MODEL}")
            print("   The model will be downloaded automatically on first use.")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def get_config_summary():
    """
    Get a summary of current configuration.
    
    Returns:
        dict: Configuration summary
    """
    return {
        'environment': ENV,
        'debug': DEBUG,
        'base_dir': str(BASE_DIR),
        'yolo_model': str(YOLO_MODEL),
        'web_host': WEB_CONFIG['HOST'],
        'web_port': WEB_CONFIG['PORT'],
        'max_file_size': f"{WEB_CONFIG['MAX_CONTENT_LENGTH'] // (1024*1024)}MB",
        'default_confidence': DEFAULT_CONFIDENCE_THRESHOLD,
        'logging_level': LOGGING_CONFIG['LEVEL']
    }

# Validate configuration on import
if __name__ != "__main__":
    validate_configuration()
