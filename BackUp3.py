import cv2
import numpy as np
from pyfirmata import Arduino, util
import tensorflow as tf
import tensorflow_hub as hub

# Initialize the Arduino board
board = Arduino('COM5')  # Update with your Arduino port
pins = {
    0: board.get_pin('d:13:o'),  # Point 0.1
    1: board.get_pin('d:12:o'),  # Point 0.2
    2: board.get_pin('d:11:o'),  # Point 0.3
    3: board.get_pin('d:10:o')   # Point 0.4
}

# Function to turn on switches based on detected points
def turn_on_switches(nearest_points):
    for i in range(4):
        pins[i].write(1 if i in nearest_points else 0)

# Load a pre-trained SSD MobileNet V2 model from TensorFlow Hub
model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(model_url)

# Preprocess input frames
def preprocess_image(frame, input_size=(300, 300)):
    resized_frame = cv2.resize(frame, input_size)
    normalized_frame = resized_frame / 255.0
    input_tensor = np.expand_dims(normalized_frame, axis=0).astype(np.float32)
    return input_tensor

# Parse detection results
def parse_results(result, frame_shape):
    height, width = frame_shape[:2]
    boxes = result['detection_boxes'][0].numpy()
    scores = result['detection_scores'][0].numpy()
    classes = result['detection_classes'][0].numpy().astype(int)

    # Define the four corner points (assuming a 640x480 resolution for simplicity)
    corner_points = [
        (480, 26),        # Top-left corner
        (34, 366),         # Top-right corner
        (934, 332),       # Bottom-left corner
        (479, 638)         # Bottom-right corner
    ]

    nearest_points = set()
    for i, score in enumerate(scores):
        if score > 0.5 and classes[i] == 1:  # Class 1 is 'person' in COCO dataset
            box = boxes[i]
            y_min, x_min, y_max, x_max = box
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)

            # Calculate the center of the bounding box
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            # Calculate distances to each corner point
            distances = [np.linalg.norm(np.array([x_center, y_center]) - np.array(point)) for point in corner_points]
            nearest_point_index = np.argmin(distances)
            nearest_points.add(nearest_point_index)

            # Draw bounding box and label
            label = f'Person {score:.2f}'
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Nearest Point: {nearest_point_index + 1}', (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.circle(frame, corner_points[nearest_point_index], 10, (0, 255, 0), -1)

    return nearest_points

# Open video capture
cap = cv2.VideoCapture(0)
print(f"Frame Width: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}, Frame Height: {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Preprocess the frame
    input_tensor = preprocess_image(frame)

    # Detect objects
    result = detector(input_tensor)

    # Parse results and determine nearest points
    nearest_points = parse_results(result, frame.shape)

    # Turn on switches based on nearest points
    turn_on_switches(nearest_points)

    # Display the image
    cv2.imshow('Object Detection with Nearest Corner Point', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()