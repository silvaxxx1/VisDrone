import cv2
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES
import supervision as sv
from PIL import Image
import numpy as np

# Load model
model = RFDETRBase()

# Load video
video_path = "demo.MP4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output writer
out = cv2.VideoWriter(
    "output_with_detections.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

# Define annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Frame-by-frame inference
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Predict
    detections = model.predict(image_pil, threshold=0.5)

    # Generate labels
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Annotate
    annotated_image = image_pil.copy()
    annotated_image = box_annotator.annotate(annotated_image, detections)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels)

    # Convert annotated PIL back to OpenCV format
    annotated_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)

    # Write frame to output video
    out.write(annotated_frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video saved as: output_with_detections.mp4")
