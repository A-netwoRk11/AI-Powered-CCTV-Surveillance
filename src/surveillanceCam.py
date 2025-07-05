import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *

# Configuration constants
CONFIDENCE = 0.4
FONT_SCALE = 1
THICKNESS = 1
DEFAULT_SKIP_FRAMES = 2
PROGRESS_UPDATE_INTERVAL = 50
MAX_PERSON_SCREENSHOTS = 3
YOLO_IMAGE_SIZE = 640

# Load YOLO model and labels
model = None
labels = []
colors = []

def initialize_surveillance_model():
    """Initialize YOLO model and labels with proper error handling."""
    global model, labels, colors
    
    if model is not None:
        return True
        
    try:
        print("[INFO] Starting surveillance model initialization...")
        
        # Always use the lightweight downloadable model for reliability
        print("[INFO] Loading YOLOv8 nano model (will download if needed)...")
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("[OK] YOLO model loaded successfully")
        
        # Try to load COCO names, fallback to default if not found
        try:
            if COCO_NAMES.exists():
                labels = open(str(COCO_NAMES)).read().strip().split("\n")
                print(f"[OK] Loaded {len(labels)} COCO class labels from file")
            else:
                raise FileNotFoundError("COCO names file not found")
        except Exception as e:
            print(f"[WARNING] Could not load COCO names file: {e}")
            print("[INFO] Using default COCO labels")
            # Comprehensive default COCO labels
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
        
        colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
        print(f"[OK] Model initialization complete - {len(labels)} classes available")
        return True
            
    except Exception as e:
        print(f"[ERROR] Error loading YOLO model: {e}")
        model = None
        labels = []
        colors = []
        return False

def draw_detection_box(image, xmin, ymin, xmax, ymax, class_id, confidence, object_name):
    """Draw detection box and label on image."""
    color = [int(c) for c in colors[class_id]]
    
    # Draw bounding box
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=THICKNESS)
    
    # Prepare text
    text = f"{object_name}: {confidence:.2f}"
    (text_width, text_height) = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, thickness=THICKNESS)[0]
    
    # Calculate text position
    text_offset_x = xmin
    text_offset_y = ymin - 5
    box_coords = ((text_offset_x, text_offset_y),
                  (text_offset_x + text_width + 2, text_offset_y - text_height))
    
    # Draw text background
    overlay = image.copy()
    cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
    image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
    
    # Draw text
    cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=FONT_SCALE, color=(0, 0, 0), thickness=THICKNESS)
    
    return image

def process_video(video_file_path, output_dir=None, skip_frames=DEFAULT_SKIP_FRAMES):
    """
    Process video file for surveillance analysis with object detection.
    
    Args:
        video_file_path: Path to input video file
        output_dir: Directory to save analyzed video (default: OUTPUT_VIDEOS_DIR)
        skip_frames: Process every Nth frame for performance (default: 2)
    
    Returns:
        dict: Analysis results including detections and output file path
    """
    print(f"[INFO] Starting surveillance analysis on: {video_file_path}")
    
    # Initialize model if not already done
    if not initialize_surveillance_model():
        print(f"[ERROR] Failed to initialize YOLO model")
        return None
    
    if output_dir is None:
        output_dir = OUTPUT_VIDEOS_DIR
    else:
        output_dir = Path(output_dir)
    
    # Create output directory with better error handling
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[WARNING] Could not create output directory {output_dir}: {e}")
        # Try temp directory as fallback
        import tempfile
        output_dir = Path(tempfile.gettempdir())
        print(f"[INFO] Using temp directory for output: {output_dir}")
    
    cap = None
    out = None
    
    try:
        cap = cv2.VideoCapture(str(video_file_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_file_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_rate <= 0:
            frame_rate = 25  # Default fallback
        
        print(f"📊 Video info: {frame_width}x{frame_height}, {frame_rate}fps, {total_frames} frames")
        
        # Generate unique output filename
        video_name = Path(video_file_path).stem
        output_num = 1
        output_filename = output_dir / f'{video_name}_analyzed_{output_num}.mp4'
        while output_filename.exists():
            output_num += 1
            output_filename = output_dir / f'{video_name}_analyzed_{output_num}.mp4'

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Better compatibility than XVID
        out = cv2.VideoWriter(str(output_filename), fourcc, frame_rate,
                              (frame_width, frame_height))
        
        if not out.isOpened():
            raise ValueError("Could not initialize video writer")
        
        # Initialize detection tracking
        detection_results = {
            'total_frames': 0,
            'detections': {},
            'person_detected': 0,
            'screenshot_saved': False,
            'output_file': str(output_filename),
            'objects_found': [],
            'processing_time': 0
        }
        
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        while True:
            ret, image = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            detection_results['total_frames'] = frame_count
            
            # Skip frames for performance
            if frame_count % skip_frames != 0:
                # Add basic info to skipped frames
                fps = f"FPS: {frame_rate:.2f}"
                cv2.putText(image, fps, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
                frame_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(image, frame_text, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(image)
                continue
                
            processed_count += 1
            
            # Progress reporting
            if processed_count % PROGRESS_UPDATE_INTERVAL == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"⏳ Processing... {frame_count}/{total_frames} frames ({progress:.1f}%) - Processed: {processed_count}")
            
            # Run YOLO detection
            start = time.perf_counter()
            results = model.predict(image, conf=CONFIDENCE, imgsz=YOLO_IMAGE_SIZE)[0]
            time_took = time.perf_counter() - start
            
            # Process detections
            for data in results.boxes.data.tolist():
                xmin, ymin, xmax, ymax, confidence, class_id = data
                xmin, ymin, xmax, ymax, class_id = int(xmin), int(ymin), int(xmax), int(ymax), int(class_id)
                
                object_name = labels[class_id] if class_id < len(labels) else f"unknown_{class_id}"
                
                # Update detection counts
                if object_name not in detection_results['detections']:
                    detection_results['detections'][object_name] = 0
                    detection_results['objects_found'].append(object_name)
                detection_results['detections'][object_name] += 1
                
                # Special handling for person detection
                if object_name == "person":
                    detection_results['person_detected'] += 1
                    if (detection_results['person_detected'] <= MAX_PERSON_SCREENSHOTS and 
                        not detection_results['screenshot_saved']):
                        
                        # Save screenshot
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        screenshot_path = SCREENSHOTS_DIR / f'person_detected_{timestamp}_{video_name}_{frame_count}.jpg'
                        os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
                        screenshot = image.copy()
                        cv2.imwrite(str(screenshot_path), screenshot)
                        detection_results['screenshot_saved'] = True
                        print(f"📸 Person detected! Screenshot saved: {screenshot_path}")
                
                # Draw detection box
                image = draw_detection_box(image, xmin, ymin, xmax, ymax, class_id, confidence, object_name)
            
            # Add frame info
            fps = f"FPS: {frame_rate:.2f}"
            cv2.putText(image, fps, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
            
            frame_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(image, frame_text, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(image)

        # Calculate processing time
        detection_results['processing_time'] = time.time() - start_time
        
        print(f"[OK] Analysis complete!")
        print(f"[INFO] Total frames processed: {detection_results['total_frames']}")
        print(f"[INFO] Persons detected: {detection_results['person_detected']}")
        print(f"[INFO] Objects found: {', '.join(detection_results['objects_found']) if detection_results['objects_found'] else 'None'}")
        print(f"[INFO] Processing time: {detection_results['processing_time']:.2f} seconds")
        print(f"[INFO] Output saved: {output_filename}")
        
        return detection_results
        
    except Exception as e:
        print(f"[ERROR] Error during video processing: {str(e)}")
        return None
        
    finally:
        # Cleanup resources
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run surveillance analysis."""
    try:
        demo_video = DEMO_VIDEOS_DIR / 'lowLightStreet.mp4'
        
        if len(sys.argv) > 1:
            video_path = Path(sys.argv[1])
            if not video_path.exists():
                print(f"[ERROR] Video file not found: {video_path}")
                return 1
        else:
            video_path = demo_video
            if not video_path.exists():
                print(f"[ERROR] Demo video not found: {video_path}")
                print("Please provide a video file path as argument:")
                print("python surveillanceCam.py path/to/video.mp4")
                return 1
        
        print("[INFO] AI-Powered Video Surveillance Analysis")
        print("=" * 50)
        
        results = process_video(video_path)
        
        if results:
            print("\n[INFO] ANALYSIS SUMMARY")
            print("=" * 30)
            for obj, count in results['detections'].items():
                print(f"  {obj}: {count} detections")
            return 0
        else:
            print("[ERROR] Analysis failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
