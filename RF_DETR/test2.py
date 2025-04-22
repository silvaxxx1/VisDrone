import cv2
import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import time

model = RFDETRBase()

cap = cv2.VideoCapture(0)
frame_count = 0
max_frames = 100  # Limit the number of frames to process

while frame_count < max_frames:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Predict using the properly converted frame
    detections = model.predict(rgb_frame, threshold=0.5)
    
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)

    # Save the frame instead of displaying it
    cv2.imwrite(f"output_frame_{frame_count:03d}.jpg", annotated_frame)
    
    frame_count += 1
    time.sleep(0.1)  # To avoid creating too many files too quickly

cap.release()
print(f"Saved {frame_count} frames to disk.")