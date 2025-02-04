# MediaPipe OSC Bridge for TouchDesigner

This project provides a bridge between MediaPipe's pose, face, and hand tracking capabilities and TouchDesigner using OSC (Open Sound Control) protocol. It captures video input from a camera, processes it through MediaPipe for skeletal tracking, and sends the data to TouchDesigner in real-time.

## Features
- Real-time pose estimation
- Face landmark detection
- Hand tracking
- OSC communication with TouchDesigner

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- python-osc
- TouchDesigner

## Installation
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the Python script:
```bash
python mediapipe_bridge.py --camera-id 0
python mediapipe_bridge.py --use-kinect --show-video
```

2. In TouchDesigner, create an OSC In CHOP to receive the data (default port: 7400)

## OSC Message Format
The script sends OSC messages in the following format:
- Pose landmarks: `/pose/point_{index} x y z`
- Face landmarks: `/face/point_{index} x y z`
- Hand landmarks: `/hand_{left/right}/point_{index} x y z` 

In TouchDesigner, create an OSC In CHOP with port 7400 to receive the data.
The script provides visual feedback through OpenCV window so you can see what's being tracked. Press ESC to quit the application.
