import torch
import cv2
import numpy as np

# Load YOLOv5 model (pre-trained on COCO or train on your own bee dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Class index for bees (customize if you train on your own dataset)
BEE_CLASS_IDX = 0  # Replace with bee's class index from your custom model if needed

# Initialize a dictionary to store trackers for each bee
trackers = {}

def detect_bees(frame):
    """
    Detect bees in the current frame using the YOLO model.
    Returns a list of bounding boxes for detected bees.
    """
    # Perform detection on the frame
    results = model(frame)
    # Convert detection results to a pandas DataFrame
    detections = results.pandas().xyxy[0]  
    bee_boxes = []

    for _, row in detections.iterrows():
        # Check if the detected class is a bee (BEE_CLASS_IDX)
        if row['class'] == BEE_CLASS_IDX:
            x1, y1, x2, y2, conf = row['xmin'], row['ymin'], row['xmax'], row['ymax'], row['confidence']
            bee_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))

    return bee_boxes

def create_tracker(frame, bbox):
    """
    Create a new CSRT tracker for a given bounding box.
    """
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    return tracker

# Initialize camera or video source
cap = cv2.VideoCapture(0)  # Replace '0' with the path to a video file if needed

bee_id = 0  # ID for each bee to track

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect bees in the frame
    bee_boxes = detect_bees(frame)

    # Create new trackers for each detected bee if necessary
    for (x1, y1, x2, y2, conf) in bee_boxes:
        # Convert bounding box to format (x, y, w, h) for tracker initialization
        bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Assign a unique ID to each bee and create a new tracker
        trackers[bee_id] = create_tracker(frame, bbox)
        bee_id += 1

    # Update and display each tracker
    for bee_id, tracker in list(trackers.items()):
        success, bbox = tracker.update(frame)
        if success:
            # Draw the updated tracking box
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'Bee {bee_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            # Remove the tracker if it fails
            del trackers[bee_id]

    # Display the frame
    cv2.imshow('Bee Detection and Tracking', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

