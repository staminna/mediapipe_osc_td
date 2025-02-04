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
This update makes several improvements:
Adds descriptive names for each hand landmark (e.g., 'index_tip', 'thumb_tip', etc.)
Sends individual OSC messages for each landmark with clear addressing (e.g., /hand/left/index_tip)
Also sends a complete hand data message at /hand/left/all or /hand/right/all for efficiency
Fixes the logging error by removing any incorrect logging.print calls
Now in TouchDesigner, you can receive messages with addresses like:
/hand/left/index_tip
/hand/left/thumb_tip
/hand/right/wrist
etc.
Each message contains an array of [x, y, z] coordinates for that specific landmark. You can also use the /hand/left/all or /hand/right/all addresses to receive all landmarks for a hand in a single message.
This should make it much easier to parse and use the data in TouchDesigner, as you can directly reference specific fingers or joints by their descriptive names.

In TouchDesigner, create an OSC In CHOP with port 7400 to receive the data.
The script provides visual feedback through OpenCV window so you can see what's being tracked. Press ESC to quit the application.
