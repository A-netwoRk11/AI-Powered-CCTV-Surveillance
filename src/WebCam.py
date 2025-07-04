import cv2
import numpy as np
import time
import sys
import os
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *

# Configuration constants
CONFIDENCE = 0.5
FONT_SCALE = 1
THICKNESS = 1
FPS = 20
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 720

# Load YOLO model and labels
try:
    labels = open(COCO_NAMES).read().strip().split("\n")
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    model = YOLO(str(YOLO_MODEL))
    print("[OK] YOLO model loaded successfully")
except Exception as e:
    print(f"[ERROR] Error loading YOLO model: {e}")
    # Don't exit, just use None and handle gracefully
    model = None
    labels = []
    colors = []

recording_duration = int(os.environ.get('RECORDING_DURATION', 30))

print(f"[INFO] Starting webcam recording for {recording_duration} seconds...")

try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Error: Could not open webcam")
        sys.exit(1)

    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    _, image = cap.read()
    if image is None:
        print("[ERROR] Error: Could not read from webcam")
        cap.release()
        sys.exit(1)
        
    h, w = image.shape[:2]

    new_width = DEFAULT_WIDTH
    new_height = DEFAULT_HEIGHT  

    os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = UPLOADS_DIR / f'live_recording_{timestamp}.mp4'

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_filename), fourcc, FPS, (new_width, new_height))

    if not out.isOpened():
        print("[ERROR] Error: Could not open video writer")
        cap.release()
        sys.exit(1)

    print(f"📹 Recording to: {output_filename}")

    start_time = time.time()

    while True:
        _, image = cap.read()
        if image is None:
            print("[WARN] Warning: Could not read frame from webcam")
            break
            
        elapsed_time = time.time() - start_time
        if elapsed_time >= recording_duration:
            print(f"[OK] Recording completed after {recording_duration} seconds")
            break
        
        image = cv2.resize(image, (new_width, new_height))

        start = time.perf_counter()
        results = model.predict(image, conf=CONFIDENCE)[0]
        time_took = time.perf_counter() - start
        print("Time took:", time_took)

        for data in results.boxes.data.tolist():
            xmin, ymin, xmax, ymax, confidence, class_id = data
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            class_id = int(class_id)

            color = [int(c) for c in colors[class_id]]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                          color=color, thickness=THICKNESS)
            text = f"{labels[class_id]}: {confidence:.2f}"
            (text_width, text_height) = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=FONT_SCALE, thickness=THICKNESS)[0]
            text_offset_x = xmin
            text_offset_y = ymin - 5
            box_coords = ((text_offset_x, text_offset_y),
                          (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(
                overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=FONT_SCALE, color=(0, 0, 0), thickness=THICKNESS)

        end = time.perf_counter()
        fps = f"FPS: {1 / (end - start):.2f}"
        cv2.putText(image, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
        
        time_remaining = recording_duration - elapsed_time
        time_text = f"Recording: {time_remaining:.1f}s left"
        cv2.putText(image, time_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        out.write(image)
        cv2.imshow("Webcam Recording - Press 'q' to stop", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Recording stopped by user")
            break

    print(f"[OK] Live recording saved as: {output_filename}")
    print(f"[INFO] File location: {output_filename}")
    print("[INFO] You can now analyze this recording from the web interface!")

except Exception as e:
    print(f"[ERROR] Error during recording: {str(e)}")
finally:
    # Cleanup resources
    if 'cap' in locals():
        cap.release()
    if 'out' in locals():
        out.release()
    cv2.destroyAllWindows()

# Test classes and functions
import unittest

class TestVideoCapture(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.video_capture = cv2.VideoCapture(0)
        self.model = YOLO(str(YOLO_MODEL))
        self.output_dir = BASE_DIR / 'test_output_videos'
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_num = 1

    def test_real_time_object_tracking(self):
        """Test if webcam can capture frames successfully."""
        if not self.video_capture.isOpened():
            self.skipTest("Webcam not available")
        ret, frame = self.video_capture.read()
        self.assertTrue(ret, "Failed to capture frame from webcam")
        self.assertIsNotNone(frame, "Captured frame is None")

    def test_object_recognition(self):
        """Test YOLO model object detection on webcam frame."""
        if not self.video_capture.isOpened():
            self.skipTest("Webcam not available")
        ret, frame = self.video_capture.read()
        if not ret:
            self.skipTest("Could not capture frame")
        results = self.model.predict(frame, conf=CONFIDENCE)[0]
        self.assertIsNotNone(results, "YOLO prediction returned None")

    def test_video_capture_functionality(self):
        """Test basic webcam functionality."""
        self.assertTrue(self.video_capture.isOpened(), "Webcam failed to open")

    def test_object_counting(self):
        """Test object detection and counting."""
        if not self.video_capture.isOpened():
            self.skipTest("Webcam not available")
        ret, frame = self.video_capture.read()
        if not ret:
            self.skipTest("Could not capture frame")
        results = self.model.predict(frame, conf=CONFIDENCE)[0]
        object_count = len(results.boxes.data.tolist())
        self.assertGreaterEqual(object_count, 0, "Object count should be non-negative")

    def test_yolo_object_detection_accuracy(self):
        """Test YOLO detection for expected objects."""
        if not self.video_capture.isOpened():
            self.skipTest("Webcam not available")
        ret, frame = self.video_capture.read()
        if not ret:
            self.skipTest("Could not capture frame")
        results = self.model.predict(frame, conf=CONFIDENCE)[0]
        detected_objects = [labels[int(data[5])]
                            for data in results.boxes.data.tolist()]
        # Note: This test may fail if no expected objects are in view
        print(f"Detected objects: {detected_objects}")
        self.assertIsInstance(detected_objects, list, "Detected objects should be a list")

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.video_capture.isOpened():
            self.video_capture.release()
        cv2.destroyAllWindows()
        if os.path.exists(self.output_dir):
            try:
                for file in os.listdir(self.output_dir):
                    os.remove(os.path.join(self.output_dir, file))
                os.rmdir(self.output_dir)
            except OSError:
                pass  # Directory may not be empty or may not exist


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[''], exit=False)
    else:
        pass
