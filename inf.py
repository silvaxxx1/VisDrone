from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load the YOLOv11n custom-trained model
model = YOLO("Vis.pt")

# Load input video
video_path = "demo.MP4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Define the output video writer
output_path = "output_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Try showing a dummy window to check if imshow is supported
show_gui = True
try:
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("Test", dummy)
    cv2.waitKey(1)
    cv2.destroyWindow("Test")
except:
    show_gui = False
    print("⚠️ GUI not available — running in headless mode (no imshow)")

# Inference loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Show live result if GUI is available
    if show_gui:
        cv2.imshow("YOLOv11n Live Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    # Save the frame to output video
    out.write(annotated_frame)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Inference complete. Output saved to: {os.path.abspath(output_path)}")
