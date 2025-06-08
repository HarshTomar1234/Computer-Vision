#!/usr/bin/env python3
"""
Diagnostic script to test YOLOv11 model detection
This will help us understand what the model is detecting and why tracking is failing
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def test_model_detection():
    """Test what the model actually detects"""
    print("🔍 DIAGNOSING MODEL DETECTION")
    print("=" * 50)
    
    # Load model and video
    model_path = "../object detection model/best.pt"
    video_path = "../input_video/15sec_input_720p.mp4"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    try:
        model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        # Check model info
        print(f"📊 Model info:")
        print(f"   Classes: {model.names}")
        print(f"   Number of classes: {len(model.names)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    try:
        cap = cv2.VideoCapture(video_path)
        print(f"✅ Video loaded: {video_path}")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
    except Exception as e:
        print(f"❌ Error loading video: {e}")
        return
    
    # Test detection on multiple frames
    print("\n🎯 TESTING DETECTION ON SAMPLE FRAMES:")
    print("-" * 40)
    
    confidence_levels = [0.1, 0.3, 0.5, 0.7]
    test_frames = [10, 50, 100, 150, 200]  # Test different points in video
    
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        print(f"\n📍 Frame {frame_num}:")
        
        for conf in confidence_levels:
            try:
                results = model(frame, conf=conf, verbose=False)
                
                total_detections = 0
                class_counts = {}
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            total_detections += 1
                            cls = int(box.cls[0].cpu().numpy())
                            class_name = model.names.get(cls, f"class_{cls}")
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            if class_name not in class_counts:
                                class_counts[class_name] = []
                            class_counts[class_name].append(confidence)
                
                print(f"   Conf {conf:.1f}: {total_detections} detections")
                if class_counts:
                    for class_name, confidences in class_counts.items():
                        avg_conf = np.mean(confidences)
                        print(f"      {class_name}: {len(confidences)} objects (avg conf: {avg_conf:.3f})")
                
            except Exception as e:
                print(f"   Conf {conf:.1f}: Error - {e}")
    
    cap.release()
    
    # Test on first frame with very low confidence and save result
    print("\n💾 SAVING DETECTION VISUALIZATION:")
    print("-" * 40)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if ret:
        try:
            # Test with very low confidence
            results = model(frame, conf=0.05, verbose=False)
            
            # Draw all detections
            annotated_frame = frame.copy()
            detection_count = 0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detection_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = model.names.get(cls, f"class_{cls}")
                        
                        # Draw bounding box
                        color = (0, 255, 0) if class_name == 'person' or 'player' in class_name.lower() else (0, 0, 255)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.3f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save annotated frame
            output_path = "detection_test.jpg"
            cv2.imwrite(output_path, annotated_frame)
            print(f"✅ Saved detection test image: {output_path}")
            print(f"   Total detections found: {detection_count}")
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
    
    cap.release()
    
    # Summary and recommendations
    print("\n🎯 DIAGNOSIS SUMMARY:")
    print("=" * 50)
    print("Check the following:")
    print("1. Look at 'detection_test.jpg' to see what the model detects")
    print("2. Verify the model was trained for the right classes")
    print("3. Check if confidence thresholds need adjustment")
    print("4. Consider if the model needs retraining for this video type")

def main():
    test_model_detection()

if __name__ == "__main__":
    main() 