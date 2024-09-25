import cv2
import torch
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # 'best.pt' is the model file

# Define a function to process video frames
def process_frame(frame):
    # Run YOLOv5 inference on the frame
    results = model(frame)

    # Extract the results (bounding boxes and confidence scores)
    bbox_data = results.xyxy[0].cpu().numpy()  # Extract the bounding box data
    for bbox in bbox_data:
        x1, y1, x2, y2, conf, cls = bbox

        # Only show bees (if class index matches bees)
        # Assuming that the model is trained with a class for bees
        if cls == 0:  # Class index for 'bee' (this should match your trained model's class index)
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Bee {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Initialize webcam or video file
# cap = cv2.VideoCapture(0)  # For webcam
cap = cv2.VideoCapture('bee_video.mp4')  # For a video file

# Video processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame to detect bees
    frame = process_frame(frame)

    # Show the processed frame
    cv2.imshow('Bee Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
