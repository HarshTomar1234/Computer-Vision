import cv2
from ultralytics import YOLO

# Load model
model = YOLO("../object detection model/best.pt")
print("Model classes:", model.names)

# Load first frame of video
cap = cv2.VideoCapture("../input_video/15sec_input_720p.mp4")
ret, frame = cap.read()

if ret:
    # Test detection with very low confidence
    results = model(frame, conf=0.01)
    
    print(f"\nDetections found:")
    for result in results:
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                print(f"  Detection {i+1}: {class_name} (class {cls}) - confidence: {conf:.3f}")
        else:
            print("  No detections found")

cap.release() 