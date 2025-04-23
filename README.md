
# Automated Power Monitoring 
 
## Overview
This project utilizes a YOLOv3-based object detection system combined with an Arduino setup to monitor human occupancy in a room and control appliances accordingly. The goal is to create an energy-efficient and smart environment that dynamically responds to the presence of people.

## Features
- Real-time human detection using YOLOv3.
- Control of appliances based on occupancy detected.
- Annotation of detected objects and nearest corner points in the video feed.
- Integration with Arduino for appliance control.

## Hardware Requirements
- **Arduino Board** (compatible with pyFirmata)
- **Digital Relay Modules** (for controlling appliances)
- **Camera** (for capturing video feed)
- **Computer** (for running the YOLOv3 model and interfacing with Arduino)
- **Connecting Cables** (for wiring the Arduino and relays)

## Software Requirements
- Python 3.x
- OpenCV
- NumPy
- PyFirmata
- YOLOv3 Configuration and Weights Files (`yolov3.cfg` and `yolov3.weights`)

## Setup Instructions

### 1. Install Python Packages
pip install opencv-python numpy pyfirmata

### 2. Arduino Setup
-Connect the relay modules to the Arduino board as per your appliance requirements.
-Update the COM port in the code to match your Arduino port.

### 3. YOLOv3 Model
-Download the YOLOv3 configuration (yolov3.cfg) and weights (yolov3.weights) files and place them in your project directory.

### 4. Define Corner Points
-Update the corner points in the code to match the actual positions in your room.

## Usage

-Run the Python Script:
-python automate_power_monitoring.py

## Monitor the Video Feed:

- The script will capture video from the camera and perform real-time human detection.
- Detected persons and nearest corner points will be annotated on the video feed.
- Appliances will be controlled based on detected occupancy.

## Code Explanation
-The main components of the code include:
-Library Imports: Import necessary libraries.
-Arduino Initialization: Initialize the Arduino board and define digital pins.
-Function to Turn On Switches: Function to control switches based on detected points.
-YOLOv3 Model Loading: Load and configure the YOLOv3 model.
-Corner Points Definition: Define the four corner points in the room.
-Video Capture Initialization: Initialize video capture from the camera.
-Main Loop for Video Processing: Process each frame to detect persons and determine the nearest corner point.
-Switch Control: Control Arduino switches based on detected nearest points.
-Displaying the Image: Display the processed video with annotations.

## Contributing
If you'd like to contribute to this project, feel free to fork the repository and submit pull requests. Any improvements or suggestions are welcome!
For any questions or support, please reach out at [raunakd511@gmail.com] & srivastavasurya0111@gmail.com

