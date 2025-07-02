<<<<<<< HEAD
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Model paths
MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
COCO_NAMES = BASE_DIR / "data" / "coco.names"

# Input/Output paths
INPUT_DIR = BASE_DIR / "input"
DEMO_VIDEOS_DIR = INPUT_DIR / "demo_videos"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_VIDEOS_DIR = OUTPUT_DIR / "videos"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
UPLOADS_DIR = OUTPUT_DIR / "uploads"

# Test paths
=======
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Model paths
MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
COCO_NAMES = BASE_DIR / "data" / "coco.names"

# Input/Output paths
INPUT_DIR = BASE_DIR / "input"
DEMO_VIDEOS_DIR = INPUT_DIR / "demo_videos"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_VIDEOS_DIR = OUTPUT_DIR / "videos"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
UPLOADS_DIR = OUTPUT_DIR / "uploads"

# Test paths
>>>>>>> 91276018eae2976736e3a9f79ca130b98400e8fb
TESTS_DIR = BASE_DIR / "tests"