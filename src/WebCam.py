import cv2
import numpy as np
import time
import sys
import os
import unittest
from ultralytics import YOLO
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Try to import config with fallback
try:
    from config.settings import *
    print("✅ Config imported in WebCam")
except Exception as e:
    print(f"⚠️ Config import failed in WebCam: {e}")
    # Fallback paths
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_VIDEOS_DIR = OUTPUT_DIR / "videos"
    SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
    UPLOADS_DIR = OUTPUT_DIR / "uploads"
    COCO_NAMES = BASE_DIR / "data" / "coco.names"
    MODELS_DIR = BASE_DIR / "models"
    YOLO_MODEL = MODELS_DIR / "yolov8n.pt"
    print(f"📂 Using fallback paths in WebCam")

CONFIDENCE = 0.5
font_scale = 1
thickness = 1

labels = open(COCO_NAMES).read().strip().split("\n") if COCO_NAMES.exists() else []
colors = np.random.randint(0, 255, size=(len(labels) if labels else 80, 3), dtype="uint8")

# Auto-download YOLO model if not found
try:
    if YOLO_MODEL.exists():
        model = YOLO(str(YOLO_MODEL))
    else:
        print("🔄 YOLO model not found locally, downloading...")
        model = YOLO('yolov8n.pt')  # Auto-downloads
        print("✅ YOLO model downloaded successfully")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = YOLO('yolov8n.pt')  # Fallback to auto-download

recording_duration = int(os.environ.get('RECORDING_DURATION', 30))

print(f"🎥 Starting webcam recording for {recording_duration} seconds...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    sys.exit(1)

_, image = cap.read()
h, w = image.shape[:2]

new_width = 1080  
new_height = 720  

os.makedirs(OUTPUT_VIDEOS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
output_filename = UPLOADS_DIR / f'live_recording_{timestamp}.mp4'

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(output_filename), fourcc, 20.0, (new_width, new_height))

print(f"📹 Recording to: {output_filename}")

start_time = time.time()

while True:
    _, image = cap.read()
    if image is None:
        break
    elapsed_time = time.time() - start_time
    if elapsed_time >= recording_duration:
        print(f"✅ Recording completed after {recording_duration} seconds")
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
                      color=color, thickness=thickness)
        text = f"{labels[class_id]}: {confidence:.2f}"
        (text_width, text_height) = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = xmin
        text_offset_y = ymin - 5
        box_coords = ((text_offset_x, text_offset_y),
                      (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(
            overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

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

print(f"✅ Live recording saved as: {output_filename}")
print(f"📁 File location: {output_filename}")
print("🔍 You can now analyze this recording from the web interface!")
cap.release()
out.release()
cv2.destroyAllWindows()

cap.release()
out.release()
cv2.destroyAllWindows()

class TestVideoCapture(unittest.TestCase):
    def setUp(self):
        self.video_capture = cv2.VideoCapture(0)
        self.model = YOLO(str(YOLO_MODEL))
        self.output_dir = BASE_DIR / 'test_output_videos'
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_num = 1

    def test_real_time_object_tracking(self):
        ret, frame = self.video_capture.read()
        self.assertTrue(ret)

    def test_object_recognition(self):
        ret, frame = self.video_capture.read()
        results = self.model.predict(frame, conf=0.5)[0]
        self.assertGreater(len(results.boxes.data.tolist()), 0)

    def test_video_capture_functionality(self):
        self.assertTrue(self.video_capture.isOpened())

    def test_object_counting(self):
        ret, frame = self.video_capture.read()
        results = self.model.predict(frame, conf=0.5)[0]
        object_count = len(results.boxes.data.tolist())
        self.assertGreater(object_count, 0)

    def test_yolo_object_detection_accuracy(self):
        ret, frame = self.video_capture.read()
        results = self.model.predict(frame, conf=0.5)[0]
        detected_objects = [labels[int(data[5])]
                            for data in results.boxes.data.tolist()]
        expected_objects = ["person"]
        for obj in expected_objects:
            self.assertIn(obj, detected_objects)

    def tearDown(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=[''], exit=False)
    else:
        pass
