import serial
import time

# port for accessing RPI 2040 on linux
LINUX_PORT = "/dev/ttyACM0"
# port for mac
MACOS_PORT = "/dev/cu.usbmodem2101"

ser = serial.Serial(LINUX_PORT, 115200)

while True:
	message = "drive\n"
	ser.write(message.encode())
	print(f"sent message: {message}")
	time.sleep(10)

	message = "turn left\n"
	ser.write(message.encode())
	print(f"sent message: {message}")
	time.sleep(10)

	message = "turn right\n"
	ser.write(message.encode())
	print(f"sent message: {message}")
	time.sleep(10)
