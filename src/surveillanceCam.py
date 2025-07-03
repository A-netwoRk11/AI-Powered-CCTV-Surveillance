import cv2
import numpy as np
import time
import sys
import os
from pathlib import Path

from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.settings import *

CONFIDENCE = 0.4
font_scale = 1
thickness = 1

labels = open(str(COCO_NAMES)).read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

model = YOLO(str(YOLO_MODEL))

def process_video(video_file_path, output_dir=None, skip_frames=2):
    print(f"🎬 Starting surveillance analysis on: {video_file_path}")
    
    if output_dir is None:
        output_dir = OUTPUT_VIDEOS_DIR
    else:
        output_dir = Path(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_file_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_file_path}")
        return None
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = int(cap.get(5))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video info: {frame_width}x{frame_height}, {frame_rate}fps, {total_frames} frames")
    
    video_name = Path(video_file_path).stem
    output_num = 1
    output_filename = output_dir / f'{video_name}_analyzed_{output_num}.mp4'
    while output_filename.exists():
        output_num += 1
        output_filename = output_dir / f'{video_name}_analyzed_{output_num}.mp4'

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(output_filename), fourcc, frame_rate,
                          (frame_width, frame_height))
    
    detection_results = {
        'total_frames': 0,
        'detections': {},
        'person_detected': 0,
        'screenshot_saved': False,
        'output_file': str(output_filename),
        'objects_found': []
    }
    
    frame_count = 0
    processed_count = 0
    
    while True:
        ret, image = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        detection_results['total_frames'] = frame_count
        
        if frame_count % skip_frames != 0:
            fps = f"FPS: {frame_rate:.2f}"
            cv2.putText(image, fps, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
            frame_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(image, frame_text, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(image)
            continue
            
        processed_count += 1
        
        if processed_count % 50 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"⏳ Processing... {frame_count}/{total_frames} frames ({progress:.1f}%) - Processed: {processed_count}")
        
        start = time.perf_counter()
        results = model.predict(image, conf=CONFIDENCE, imgsz=640)[0]  # Smaller image size
        time_took = time.perf_counter() - start
        
        for data in results.boxes.data.tolist():
            xmin, ymin, xmax, ymax, confidence, class_id = data
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            class_id = int(class_id)
            
            object_name = labels[class_id] if class_id < len(labels) else f"unknown_{class_id}"
            
            if object_name not in detection_results['detections']:
                detection_results['detections'][object_name] = 0
                detection_results['objects_found'].append(object_name)
            detection_results['detections'][object_name] += 1
            
            if object_name == "person":
                detection_results['person_detected'] += 1
                if detection_results['person_detected'] <= 3 and not detection_results['screenshot_saved']:
                    text = f"{object_name}"
                    color = [int(c) for c in colors[class_id]]
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                                  color=color, thickness=thickness)
                    cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale, color=color, thickness=thickness)
                    
                    screenshot_path = SCREENSHOTS_DIR / f'person_detected_{video_name}_{frame_count}.jpg'
                    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
                    screenshot = image.copy()
                    cv2.imwrite(str(screenshot_path), screenshot)
                    detection_results['screenshot_saved'] = True
                    print(f"📸 Person detected! Screenshot saved: {screenshot_path}")
            
            color = [int(c) for c in colors[class_id]]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                          color=color, thickness=thickness)
            text = f"{object_name}: {confidence:.2f}"
            
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
        
        fps = f"FPS: {frame_rate:.2f}"
        cv2.putText(image, fps, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
        
        frame_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(image, frame_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Analysis complete!")
    print(f"📊 Total frames processed: {detection_results['total_frames']}")
    print(f"👥 Persons detected: {detection_results['person_detected']}")
    print(f"🎯 Objects found: {', '.join(detection_results['objects_found']) if detection_results['objects_found'] else 'None'}")
    print(f"💾 Output saved: {output_filename}")
    
    return detection_results

def main():
    demo_video = DEMO_VIDEOS_DIR / 'lowLightStreet.mp4'
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return
    else:
        video_path = demo_video
        if not video_path.exists():
            print(f"❌ Demo video not found: {video_path}")
            print("Please provide a video file path as argument:")
            print("python surveillanceCam.py path/to/video.mp4")
            return
    
    print("🎥 AI-Powered Video Surveillance Analysis")
    print("=" * 50)
    
    results = process_video(video_path)
    
    if results:
        print("\n📈 ANALYSIS SUMMARY")
        print("=" * 30)
        for obj, count in results['detections'].items():
            print(f"  {obj}: {count} detections")

if __name__ == "__main__":
    main()
