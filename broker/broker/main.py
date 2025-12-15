from io import BytesIO
from pathlib import Path
import random
import subprocess
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
    camera.configure(camera.create_still_configuration())
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
                    )
                    if response.status_code == 200:
                        data = response.content
                        ser.write(data)
                        print(f"[green]✓[/green] Frame sent, wrote {len(data)} bytes to serial")
                    else:
                        print(f"[yellow]![/yellow] Server responded: {response.status_code}")
                except Exception as e:
                    print(f"[red]✗[/red] Error: {e}")
                time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n[yellow]Shutting down...[/yellow]")
    finally:
        camera.stop()
        ser.close()
        print("[green]✓[/green] Done")


@app.command()
def deploy(
    server: Annotated[
        str,
        typer.Argument(
            help="Server to deploy to (e.g., user@hostname or IP address)",
            envvar="DEPLOY_SERVER",
        ),
    ] = None,
):
    """
    Deploy Broker to a remote server.

    Example:\n
    broker deploy user@192.168.1.100\n
    broker deploy root@my-server.com\n\n

    Or set DEPLOY_SERVER environment variable:\n
    export DEPLOY_SERVER=user@server.com\n
    broker deploy\n
    """
    if not server:
        print("[red]Error: Server not specified[/red]")
        print("Usage: cb deploy user@server")
        print("Or set: export DEPLOY_SERVER=user@server")
        raise typer.Exit(1)
    print(f"[blue]→[/blue] Deploying to: {server}")
    # Check if we're in the project root
    if not Path("pyproject.toml").exists():
        print("[red]Error: Must run from project root (where pyproject.toml is)[/red]")
        raise typer.Exit(1)
    # Build the wheel
    print("[blue]→[/blue] Building wheel...")
    result = subprocess.run(["uv", "build"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[red]Build failed:[/red]")
        print(result.stderr)
        raise typer.Exit(1)
    # Find the latest wheel
    wheels = sorted(Path("dist").glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        print("[red]No wheel found in dist/[/red]")
        raise typer.Exit(1)
    wheel = wheels[-1]
    print(f":white_check_mark: Built: {wheel.name}")
    # Copy to server
    print(f"[blue]→[/blue] Copying to {server}...")
    result = subprocess.run(["scp", str(wheel), f"{server}:/tmp/"], capture_output=True)
    if result.returncode != 0:
        print("[red]Failed to copy to server[/red]")
        print(result.stderr.decode())
        raise typer.Exit(1)
    print(":white_check_mark: Copied to server")
    # Install on server
    print("[blue]→[/blue] Installing on server...")
    install_script = f"""
set -euo pipefail

# Add common uv installation paths to PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin:$PATH"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify uv is now available
if ! command -v uv &> /dev/null; then
    echo "✗ Error: uv not found in PATH"
    echo "PATH=$PATH"
    exit 1
fi

# Install the wheel
WHEEL=/tmp/{wheel.name}
echo "Installing: $WHEEL"
uv tool install --force "$WHEEL"

# Ensure PATH is set in bashrc for future logins
if ! grep -q '.local/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Test installation
if command -v cb &> /dev/null; then
    echo "✓ Installation successful!"
    cb --version 2>/dev/null || (cb --help | head -n 3) || true
else
    echo "✗ Warning: cb not in PATH. Re-login or source ~/.bashrc"
fi

# Cleanup
rm -f /tmp/{wheel.name}
"""
    result = subprocess.run(
        ["ssh", server, "bash"],
        input=install_script,
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        print(f"[red]Installation script exited with code: {result.returncode}[/red]")
        raise typer.Exit(1)
    print("\n:white_check_mark: [green]Deployment complete![/green]")
    print(f"\nConnect: [blue]ssh {server}[/blue]")
    print("Then run: [blue]broker --help[/blue]")

