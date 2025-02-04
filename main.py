try:
    # Try different possible import names
    try:
        import freenect2
    except ImportError:
        from freenect2 import PyFreenect2
        freenect2 = PyFreenect2()
    
    import sys
    import time
    
    # Initialize the library
    fn = freenect2
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("No Kinect devices found!")
        sys.exit(1)
    
    serial = fn.getDefaultDeviceSerialNumber()
    device = fn.openDevice(serial)
    
    # Add error checking
    if device is None:
        print("Failed to open device")
        sys.exit(1)
    
    # Add small delay after device start
    time.sleep(2)  # Give the device time to fully initialize
    
    while True:
        frames = device.waitForNewFrame()
        if frames is None:
            print("Failed to read frames")
            continue
            
        # Process frames here
        device.releaseFrames(frames)
        
except Exception as e:
    print(f"Error: {e}")
    if 'device' in locals():
        device.close()
finally:
    if 'device' in locals():
        device.close() 