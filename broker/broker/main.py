from io import BytesIO
import random
import time
from typing import Annotated
import typer
from rich import print
import httpx
import serial

app = typer.Typer()


@app.command()
def start(
    relay_server_address: Annotated[
        str,
        typer.Option(help="Address of the relay server", envvar="RELAY_SERVER_ADDRESS"),
    ] = "http://localhost:8000",
    serial_port: Annotated[
        str,
        typer.Option(help="Serial port", envvar="SERIAL_PORT"),
    ] = "/dev/ttyACM0",
    baud_rate: Annotated[
        int,
        typer.Option(help="Serial baud rate", envvar="BAUD_RATE"),
    ] = 115200,
    mock: Annotated[
        bool,
        typer.Option(help="Mock mode", envvar="MOCK_MODE"),
    ] = False,
    interval: Annotated[
        int,
        typer.Option(help="Capture interval in milliseconds", envvar="CAPTURE_INTERVAL"),
    ] = 5000,
):
    """
    Start the Broker.
    
    Continuously captures images from the PiCamera and sends them to the relay server.
    """
    # Import picamera2 here so the module can be imported on non-Pi systems
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("[red]Error: picamera2 not installed[/red]")
        print("Install with: sudo apt install -y libcamera-apps python3-libcamera && uv pip install picamera2")
        raise typer.Exit(1)
    print(f"[blue]→[/blue] Relay server: {relay_server_address}")
    print(f"[blue]→[/blue] Mock mode: {mock}")
    print(f"[blue]→[/blue] Serial port: {serial_port} @ {baud_rate} baud")
    print(f"[blue]→[/blue] Capture interval: {interval}ms")
    print("[blue]→[/blue] Opening serial port...")
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print("[green]✓[/green] Serial port opened")
    print("[blue]→[/blue] Initializing camera...")
    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        buffer_count=2,
    )
    camera.configure(config)
    camera.start()
    print("[green]✓[/green] Camera initialized")
    interval_seconds = interval / 1000.0
    print("[green]✓[/green] Running (Ctrl+C to stop)")
    try:
        with httpx.Client(timeout=10.0) as client:
            while True:
                try:
                    if mock:
                        mock_data = ["drive", "turn_left", "turn_right", "stop"]
                        ser.write(f"{mock_data[random.randint(0, len(mock_data) - 1)]}\n".encode())
                        print(f"[green]✓[/green] Mock data sent: {mock_data[random.randint(0, len(mock_data) - 1)]}")
                        time.sleep(interval_seconds)
                        continue
                    buffer = BytesIO()
                    camera.capture_file(buffer, format="jpeg")
                    buffer.seek(0)
                    response = client.post(
                        f"{relay_server_address}/api/v1/actions",
                        files={"file": ("frame.jpg", buffer, "image/jpeg")},
                    ).json()
                    action = f"{response['action']}\n"
                    ser.write(action.encode())
                    print(f"[green]✓[/green] Action: {response['action']}")
                except Exception as e:
                    print(f"[red]✗[/red] Error: {e}")
                time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n[yellow]Shutting down...[/yellow]")
    finally:
        camera.stop()
        ser.close()
        print("[green]✓[/green] Done")
