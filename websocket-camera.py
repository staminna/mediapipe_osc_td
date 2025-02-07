import asyncio
import json
import cv2
import mediapipe as mp
import numpy as np
from functools import partial
import websockets
import platform

# Define the new driver name for TouchDesigner
DRIVER_NAME = "Kinect v2 camera for Apple"

# For Kinect v2 on macOS (requires libfreenect2 via Homebrew)
try:
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame
    KINECT_AVAILABLE = True
except ImportError:
    KINECT_AVAILABLE = False

# Configuration
WEBSOCKET_PORT = 8765
TARGET_RESOLUTION = (640, 480)  # Optimized for MediaPipe performance
FRAME_RATE = 30

async def process_frames(websocket, use_kinect=True):
    # Send driver identification info to TouchDesigner
    await websocket.send(json.dumps({
        "type": "driver_info",
        "driver": DRIVER_NAME,
        "compatible": "M1 Pro and above",
        "resolution": TARGET_RESOLUTION,
        "fps": FRAME_RATE
    }))

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # Lower complexity for better performance
        enable_segmentation=False,
        min_detection_confidence=0.7
    )

    # Initialize video source (Kinect if available, otherwise fallback to webcam)
    if use_kinect and KINECT_AVAILABLE:
        print(f"Initializing {DRIVER_NAME} using Kinect v2...")
        fn = Freenect2()
        device = fn.openDevice(fn.getDefaultDeviceSerialNumber())
        listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)
        device.setColorFrameListener(listener)
        device.setIrAndDepthFrameListener(listener)
        device.start()
        registration = Registration(device.getIrCameraParams(),
                                    device.getColorCameraParams())
    else:
        # Fallback to webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_RESOLUTION[1])
        cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    try:
        while True:
            if use_kinect and KINECT_AVAILABLE:
                # Kinect frame processing
                frames = listener.waitForNewFrame()
                color_frame = frames["color"]
                frame = color_frame.asarray()  # (1080, 1920, 4)
                frame = cv2.resize(frame, TARGET_RESOLUTION)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                listener.release(frames)
            else:
                # Webcam processing
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe processing
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        "id": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })

                # Send pose data as JSON payload
                await websocket.send(json.dumps({
                    "type": "pose_data",
                    "landmarks": landmarks,
                    "resolution": TARGET_RESOLUTION,
                    "fps": FRAME_RATE
                }))

            # Maintain frame rate
            await asyncio.sleep(1 / FRAME_RATE)
    finally:
        if use_kinect and KINECT_AVAILABLE:
            device.stop()
            device.close()
        else:
            cap.release()
        pose.close()

async def main():
    try:
        if KINECT_AVAILABLE:
            print(f"Starting {DRIVER_NAME} server using Kinect v2...")
            server = await websockets.serve(partial(process_frames, use_kinect=True),
                                            "localhost", WEBSOCKET_PORT)
        else:
            raise RuntimeError("Kinect not available")
    except Exception as e:
        print(f"Kinect error: {e}, falling back to webcam")
        server = await websockets.serve(partial(process_frames, use_kinect=False),
                                        "localhost", WEBSOCKET_PORT)

    print(f"WebSocket server running on ws://localhost:{WEBSOCKET_PORT}")

    # Keep the event loop alive forever
    await asyncio.Future()  # This awaits something that never finishes


if __name__ == "__main__":

    # Configure for Apple Silicon (M1 Pro and above) performance on macOS
    if platform.system() == "Darwin" and platform.processor() == "arm":
        try:
            from Foundation import NSBundle
            bundle = NSBundle.mainBundle()
            if bundle:
                info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
                info["LSAppNapIsDisabled"] = True
        except ImportError:
            print("Foundation module not found; skipping NSBundle configuration.")
    asyncio.run(main())
