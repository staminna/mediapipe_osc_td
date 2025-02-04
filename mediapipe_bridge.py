#!/usr/bin/env python3
"""
Kinect v2 + MediaPipe Pose Bridge for TouchDesigner (M1/M2 Optimized)
"""

import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import argparse
import sys
import signal
import time
import logging
import os
import traceback

## Conditional Kinect imports
KINECT_AVAILABLE = False
try:
    # Attempt to import the Python bindings for libfreenect2.
    # These bindings are required for using Kinect v2.
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Frame, FrameMap
    KINECT_AVAILABLE = True
except ImportError:
    # Kinect support is disabled if pylibfreenect2 cannot be imported.
    KINECT_AVAILABLE = False

# Configuration
DEFAULT_OSC_IP = "127.0.0.1"
DEFAULT_OSC_PORT = 7400
FPS_REPORT_INTERVAL = 5  # seconds

class KinectSource:
    def __init__(self):
        if not KINECT_AVAILABLE:
            raise RuntimeError("Kinect requested (--use-kinect) but pylibfreenect2 is not available. "
                                   "Please install the required Python bindings following the instructions at "
                                   "https://openkinect.github.io/libfreenect2/")
            
        self.freenect = Freenect2()
        num_devices = self.freenect.enumerateDevices()
        if num_devices == 0:
            raise RuntimeError("No Kinect devices detected")
        
        self.serial = self.freenect.getDeviceSerialNumber(0)
        self.device = self.freenect.openDevice(self.serial)
        
        self.listener = SyncMultiFrameListener(FrameType.Color)
        self.device.setColorFrameListener(self.listener)
        
        self.device.start()
        logging.info("Kinect v2 initialized | Serial: %s", self.serial)

    def get_frame(self):
        frames = FrameMap()
        if self.listener.waitForNewFrame(frames, 1000):  # 1 second timeout
            try:
                color_frame = frames[FrameType.Color]
                return cv2.cvtColor(color_frame.asarray(), cv2.COLOR_RGBA2BGR)
            finally:
                self.listener.release(frames)
        return None

    def release(self):
        if hasattr(self, 'device') and self.device:
            self.device.stop()
            self.device.close()
        logging.info("Kinect shutdown complete")

class WebcamSource:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Webcam {camera_id} unavailable")
        logging.info("Webcam initialized")

    def get_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

class PoseAndHandProcessor:
    def __init__(self, args):
        logging.info("Initializing PoseAndHandProcessor...")
        self.osc_client = udp_client.SimpleUDPClient(args.ip, args.port)
        self.send_interval = 1.0 / args.max_fps if args.max_fps > 0 else 0
        self.last_send = 0
        
        # Initialize MediaPipe solutions
        logging.info("Setting up MediaPipe components...")
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize pose with more stable settings
        logging.info("Initializing pose detector...")
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logging.info("Initializing hand detector...")
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Error tracking
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.last_error_time = 0
        self.error_timeout = 2.0  # seconds
        
        # Frame processing stats
        self.processed_frames = 0
        self.failed_frames = 0
        self.last_stats_time = time.time()
        
        # Drawing specs
        self.pose_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=2,
            circle_radius=2
        )
        self.hand_drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=2,
            circle_radius=2
        )
        
        # Initialize source
        if args.use_kinect:
            if not KINECT_AVAILABLE:
                raise RuntimeError("Kinect support not available")
            self.source = KinectSource()
        else:
            self.source = WebcamSource(args.camera_id)
        
        self.draw_landmarks = args.show_video
        self.running = True
        self.frame_count = 0
        self.last_fps_report = time.time()
        
        # Track last known good pose for stability
        self.last_good_pose = None
        self.pose_timeout = 0.5  # seconds
        self.last_pose_time = 0
        
        # Add drawing methods
        self.draw_pose = self._draw_pose  # Add reference to drawing method
        self.draw_hands = self._draw_hands  # Add reference to drawing method
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        logging.info("Shutdown initiated")
        self.running = False

    def _draw_pose(self, frame, landmarks):
        """Draw pose landmarks and connections on the frame."""
        try:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # Green color
                    thickness=2,
                    circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 0),  # Yellow color
                    thickness=2
                )
            )
            # Add landmark labels for better visualization
            for idx, landmark in enumerate(landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, str(idx), (cx, cy), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            logging.error(f"Error drawing pose: {str(e)}")

    def _draw_hands(self, frame, hand_landmarks, handedness):
        """Draw hand landmarks and connections on the frame."""
        try:
            color = (0, 0, 255) if handedness == "Right" else (255, 0, 0)
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=color,
                    thickness=2,
                    circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=color,
                    thickness=2
                )
            )
        except Exception as e:
            logging.error(f"Error drawing hand: {str(e)}")

    def process_frame(self, frame):
        try:
            if frame is None:
                logging.warning("Received empty frame")
                self.failed_frames += 1
                return None
                
            if frame.size == 0:
                logging.warning("Received zero-size frame")
                self.failed_frames += 1
                return None
            
            logging.debug(f"Processing frame {self.processed_frames + 1} - Shape: {frame.shape}")
            
            # Convert to RGB with error checking
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logging.error(f"Color conversion failed: {str(e)}")
                self.failed_frames += 1
                return frame
            
            # Process pose with error handling
            try:
                pose_results = self.pose.process(rgb_frame)
                if pose_results is None:
                    logging.warning("Pose processing returned None")
            except Exception as e:
                logging.error(f"Pose processing error: {str(e)}")
                self.consecutive_errors += 1
                if self.consecutive_errors >= self.max_consecutive_errors:
                    logging.critical("Too many consecutive errors, resetting pose detector...")
                    self.reset_pose_detector()
                return frame
            
            # Reset error counter on successful processing
            self.consecutive_errors = 0
            
            # Process and log pose results
            if pose_results and pose_results.pose_landmarks:
                logging.debug("Pose detected - Landmarks found")
                self.send_pose_osc(pose_results.pose_landmarks)
                
                if self.draw_landmarks:
                    self._draw_pose(frame, pose_results.pose_landmarks)  # Use internal method
            
            # Process hands
            hand_results = self.hands.process(rgb_frame)
            if hand_results and hand_results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    handedness = hand_results.multi_handedness[idx].classification[0].label
                    confidence = hand_results.multi_handedness[idx].classification[0].score
                    
                    if confidence > 0.7:
                        self.send_hand_osc(hand_landmarks, handedness)
                        if self.draw_landmarks:
                            self._draw_hands(frame, hand_landmarks, handedness)  # Use internal method
            
            self.processed_frames += 1
            
            # Log processing stats periodically
            current_time = time.time()
            if current_time - self.last_stats_time >= 5.0:
                self.log_processing_stats()
                self.last_stats_time = current_time
            
            return frame
            
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            traceback.print_exc()
            self.failed_frames += 1
            return frame

    def reset_pose_detector(self):
        logging.info("Resetting pose detector...")
        try:
            self.pose.close()
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.consecutive_errors = 0
            logging.info("Pose detector reset successful")
        except Exception as e:
            logging.error(f"Failed to reset pose detector: {str(e)}")

    def log_processing_stats(self):
        total_frames = self.processed_frames + self.failed_frames
        success_rate = (self.processed_frames / total_frames * 100) if total_frames > 0 else 0
        logging.info(f"Processing Stats: "
                    f"Processed: {self.processed_frames}, "
                    f"Failed: {self.failed_frames}, "
                    f"Success Rate: {success_rate:.1f}%")
        # Reset counters
        self.processed_frames = 0
        self.failed_frames = 0

    def send_pose_osc(self, landmarks):
        now = time.time()
        if (now - self.last_send) < self.send_interval:
            return
        
        try:
            # Format pose data with visibility scores
            data = []
            for idx, lm in enumerate(landmarks.landmark):
                data.extend([lm.x, lm.y, lm.z, lm.visibility])
                
            # Send basic pose data
            self.osc_client.send_message("/pose", data)
            
            # Send additional pose information
            pose_info = {
                "timestamp": now,
                "num_landmarks": len(landmarks.landmark),
                "frame_id": self.frame_count
            }
            self.osc_client.send_message("/pose/info", pose_info)
            
            logging.debug(f"Sent pose data: {len(data)/4} landmarks")
            self.last_send = now
            
        except Exception as e:
            logging.error(f"OSC Error (Pose): {str(e)}")

    def send_hand_osc(self, landmarks, handedness):
        now = time.time()
        if (now - self.last_send) < self.send_interval:
            return
            
        try:
            # Define hand landmark names for better readability
            landmark_names = [
                'wrist',
                'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
                'index_mcp', 'index_pip', 'index_dip', 'index_tip',
                'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
                'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
                'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
            ]
            
            # Send individual landmarks with descriptive addresses
            for idx, lm in enumerate(landmarks.landmark):
                if idx < len(landmark_names):
                    landmark_name = landmark_names[idx]
                    address = f"/hand/{handedness.lower()}/{landmark_name}"
                    data = [lm.x, lm.y, lm.z]
                    self.osc_client.send_message(address, data)
            
            # Also send complete hand data in one message for efficiency
            full_data = []
            for lm in landmarks.landmark:
                full_data.extend([lm.x, lm.y, lm.z])
            self.osc_client.send_message(f"/hand/{handedness.lower()}/all", full_data)
            
            logging.debug(f"Sent {handedness} hand data: {len(landmark_names)} landmarks")
            self.last_send = now
            
        except Exception as e:
            logging.error(f"OSC Error (Hand): {str(e)}")

    def run(self):
        logging.info("Starting processing loop...")
        try:
            while self.running:
                start_time = time.time()
                
                try:
                    frame = self.source.get_frame()
                    if frame is None:
                        logging.warning("No frame received from source")
                        time.sleep(0.01)
                        continue
                    
                    processed = self.process_frame(frame)
                    
                    if self.draw_landmarks and processed is not None:
                        cv2.imshow('Pose Feed', processed)
                        if cv2.waitKey(1) == 27:  # ESC
                            logging.info("ESC pressed, shutting down...")
                            break
                    
                except Exception as e:
                    logging.error(f"Processing loop error: {str(e)}")
                    traceback.print_exc()
                    time.sleep(0.1)  # Prevent rapid error loops
                    
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received")
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("Starting cleanup...")
        try:
            self.pose.close()
            self.hands.close()
            self.source.release()
            cv2.destroyAllWindows()
            logging.info("Cleanup completed successfully")
        except Exception as e:
            logging.error(f"Cleanup error: {str(e)}")
        finally:
            os._exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description="Pose and Hand Tracking Bridge")
    parser.add_argument("--ip", default=DEFAULT_OSC_IP, help="OSC target IP")
    parser.add_argument("--port", type=int, default=DEFAULT_OSC_PORT, help="OSC target port")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--use-kinect", action="store_true", help="Use Kinect v2")
    parser.add_argument("--show-video", action="store_true", help="Show processing window")
    parser.add_argument("--max-fps", type=int, default=30, help="Max OSC send rate")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0,1,2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-osc", action="store_true", default=True,
                       help="Enable OSC message logging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    try:
        processor = PoseAndHandProcessor(args)
        processor.run()
    except Exception as e:
        logging.critical("Fatal error: %s", str(e))
        traceback.print_exc()
        os._exit(1)