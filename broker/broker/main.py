from pathlib import Path
import subprocess
import typer
from rich import print

app = typer.Typer()

@app.command()
def main():
    typer.echo("Hello, World!")


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
    print("Then run: [blue]cb --help[/blue]")

