#!/usr/bin/env python3
"""
Image Receiver Server (for testing)
Receives images from Raspberry Pi and saves them
Run this on receiving computer (desktop/laptop)
"""

import socket
import sys
import os
from datetime import datetime

# Configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 8000 # make sure this is the same as whatever you set on RPi
SAVE_DIR = 'received_images'  # Directory to save images

def ensure_save_directory():
    """Create the save directory if it doesn't exist"""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"Created directory: {SAVE_DIR}")

def receive_image(connection, client_address):
    """Receive a single image from the connection"""
    try:
        print(f"Receiving image from {client_address[0]}:{client_address[1]}...")
        
        # Read all data until connection closes
        image_data = connection.read()
        
        if not image_data:
            print("No data received")
            return False
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"image_{timestamp}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)
        
        # Save the image
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        size_kb = len(image_data) / 1024
        print(f"✓ Image saved: {filename} ({size_kb:.2f} KB)")
        return True
        
    except Exception as e:
        print(f"Error receiving image: {e}")
        return False

def start_server():
    """Start the receiver server and listen for connections"""
    ensure_save_directory()
    
    server_socket = None
    
    try:
        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        
        print("=" * 50)
        print("Image Receiver Server")
        print("=" * 50)
        print(f"Listening on port {PORT}")
        print(f"Images will be saved to: {os.path.abspath(SAVE_DIR)}")
        print("Waiting for connections... (Press Ctrl+C to stop)")
        print()
        
        image_count = 0
        
        while True:
            # Accept connection
            client_socket, client_address = server_socket.accept()
            connection = client_socket.makefile('rb')
            
            try:
                if receive_image(connection, client_address):
                    image_count += 1
                    print(f"Total images received: {image_count}")
                    print()
                
            finally:
                connection.close()
                client_socket.close()
    
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        
    except socket.error as e:
        print(f"Socket error: {e}")
        if "Address already in use" in str(e):
            print(f"Port {PORT} is already in use. Try closing other programs or use a different port.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
        
    finally:
        if server_socket:
            server_socket.close()
        print("Server closed")


if __name__ == '__main__':
    start_server()