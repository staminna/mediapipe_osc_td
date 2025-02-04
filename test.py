#!/usr/bin/env python3
import cv2
import numpy as np
import time

# Create a dummy black image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
cv2.moveWindow("Test Window", 0, 0)
cv2.imshow("Test Window", img)
print("Please check if 'Test Window' appears. Waiting for 5 sec...")
cv2.waitKey(5000)
cv2.destroyAllWindows()
