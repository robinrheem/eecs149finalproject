# RPi Setup
For this project we used a RPi Zero W board. Due to its limited memory and compute, installation of headless OS is recommended (we used Raspberry Pi OS Lite 32-bit). As such you will need to ssh into the board to develop or run programs.

```
ssh rpi@rpi.local
```
In this case we set up the device with name "rpi" and username "rpi". Replace the username and device name with your own device's info. Also, you can replace "rpi.local" with the full IP address of the RPi if it's known.

# Required Packages for RPi Zero (one-time setup)
```
sudo apt update
sudo apt install -y libcamera-apps python3-libcamera python3-picamera2 python3-serial python3-pip
```

```
pip install serial python-dotenv requests
```