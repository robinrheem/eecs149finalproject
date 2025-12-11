#!/usr/bin/env python3
"""
Raspberry Pi Zero Camera Sender (picamera2 version)
Captures image with picamera2 and streams directly to receiver
Run this on Raspberry Pi Zero
"""

import socket
import io
import time
import sys
import os
from dotenv import load_dotenv
from picamera2 import Picamera2
from libcamera import Transform

load_dotenv()

# Configuration
# IMPORTANT: Add your IP and port in a .env file
RECEIVER_IP = os.getenv("IP")
RECEIVER_PORT = int(os.getenv("PORT"))
CAMERA_RESOLUTION = (720, 960)  # Adjust as needed
CAMERA_WARMUP = 2  # Num seconds to let camera warm up

def send_image():
    """Capture and send a single image to the receiver"""
    picam2 = None
    client_socket = None
    connection = None
    
    try:
        print(f"Connecting to {RECEIVER_IP}:{RECEIVER_PORT}...")
        
        # Connect to receiver
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((RECEIVER_IP, RECEIVER_PORT))
        connection = client_socket.makefile('wb')
        
        print("Connected! Initializing camera...")
        
        # Initialize and config camera for still capture
        picam2 = Picamera2()
        config = picam2.create_still_configuration(
            main={"size": CAMERA_RESOLUTION}
        )
        picam2.configure(config)
        
        # Start and warm up camera
        picam2.start()
        print(f"Warming up camera for {CAMERA_WARMUP} seconds...")
        time.sleep(CAMERA_WARMUP)
        
        # Capture to in-memory buffer
        print("Capturing and sending image...")
        stream = io.BytesIO()
        picam2.capture_file(stream, format='jpeg')
        
        # Send the image data
        stream.seek(0)
        connection.write(stream.read())
        connection.flush()
        
        print("Image sent successfully!")
        
    except socket.error as e:
        print(f"Network error: {e}")
        print("Make sure the receiver is running and the IP address is correct")
        sys.exit(1)
        
    except Exception as e:
        print(f"Camera or unexpected error: {e}")
        print("Make sure the camera is connected and libcamera is working")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        # Clean up
        if connection:
            connection.close()
        if client_socket:
            client_socket.close()
        if picam2:
            picam2.stop()
            picam2.close()
        print("Cleanup complete")


def send_continuous(interval=5):
    """Continuously capture and send images at specified interval (seconds)"""
    print(f"Starting continuous capture mode (interval: {interval}s)")
    print("Press Ctrl+C to stop")
    
    while True:
        try:
            send_image()
            print(f"Waiting {interval} seconds before next capture...")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopping continuous capture")
            break


if __name__ == '__main__':
    print("=" * 50)
    print("Raspberry Pi Camera Sender (picamera2)")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'continuous':
        # Continuous mode
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        send_continuous(interval)
    else:
        # Single shot mode
        send_image()