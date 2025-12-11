# Backend Server

Main goal of this server is to give actions for the robot to do given its view(image).
There's also a demo with a webcam browser for verification of the implementation.

## Quick Start

`uv` is required to run.

### 1. Install Dependencies

```bash
uv sync
```

### 2. Run the Project

Uses `honcho` to run both llama-server and FastAPI together:

```bash
uv run honcho start
```

This will start:
- llama-server on `http://localhost:8080`
- FastAPI web server on `http://localhost:8000`

### 3. Open the Demo

Navigate to `http://localhost:8000` in your browser and click "Start Camera"!

## Project Structure

```
backend/
├── readyaction/
│   ├── main.py              # FastAPI app and API endpoints
│   └── templates/
│       └── index.html       # Webcam demo interface
├── Procfile                 # Process definitions for honcho
├── pyproject.toml           # Dependencies
└── README.md
```

## API Endpoints

- `GET /` - Demo webcam interface
- `POST /api/v1/actions` - Analyze image and return VLM response

### Example API Usage

```python
import requests

with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/actions",
        files={"file": f}
    )
    print(response.json())
```

## Configuration

### Customize VLM Prompts

Edit `readyaction/main.py`:

```python
SYSTEM_PROMPT = """Your custom system prompt here"""
USER_PROMPT = "Your custom question about the image"
```

### Change Capture Interval

Edit `readyaction/templates/index.html`:

```javascript
// Change 500 to your desired interval in milliseconds
intervalId = setInterval(captureAndAnalyze, 500);
```

## Requirements

- Python 3.13+
- llama-server (from llama.cpp)
- Webcam
- Modern web browser with camera support

## Notes

- First run will download the SmolVLM model (~1GB)
- Make sure to grant camera permissions when prompted
- The VLM runs locally via llama-server
- GPU acceleration is used automatically if available

## Troubleshooting

**"Error accessing webcam"**
- Grant camera permissions in your browser
- Make sure no other app is using the camera

**"Connection refused to localhost:8080"**
- Ensure llama-server is running
- Check that port 8080 is not in use by another process

**"Model download is slow"**
- First run downloads the model from Hugging Face
- Subsequent runs will use the cached model

