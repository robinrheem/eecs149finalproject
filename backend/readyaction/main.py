from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Literal
from collections import deque
from ultralytics import YOLO
import base64
import time
import cv2
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key=""
)

# In-memory storage for robot state
robot_state = {
    "latest_frame": None,  # bytes
    "latest_frame_base64": None,  # for browser display
    "frame_timestamp": 0,
    "latest_result": None,
    "result_timestamp": 0,
}
command_queue: deque = deque(maxlen=100)  # Commands for robot to fetch
yolo_model: YOLO = YOLO("yolov8n.pt")

SYSTEM_PROMPT = """You are a vision system for a robot car. Analyze the image and provide a brief description of what you see.
Focus on objects, people, obstacles, and any relevant environmental details that would help the robot navigate or interact with its surroundings."""

class CommandRequest(BaseModel):
    command: str
    prompt: Optional[str] = None


def detect_objects(image_bytes: bytes, target: Optional[str] = None, conf_threshold: float = 0.5) -> dict:
    """Run YOLO object detection on image using Ultralytics YOLOv8"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"success": False, "error": "Failed to decode image"}
    results = yolo_model(img, conf=conf_threshold, verbose=False)
    detections = []
    target_found = False
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            confidence = float(box.conf[0])
            # xyxy format: [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            # Convert to [x, y, width, height] for compatibility
            x1, y1, x2, y2 = xyxy
            bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "box": bbox,  # [x, y, w, h]
                "xyxy": [int(x) for x in xyxy],  # [x1, y1, x2, y2]
            })
            if target and target.lower() in class_name.lower():
                target_found = True
    result = {
        "success": True,
        "mode": "detect",
        "detections": detections,
        "count": len(detections)
    }
    if target:
        result["target"] = target
        result["target_found"] = target_found
    return result


def reason_about_image(image_bytes: bytes, prompt: str) -> dict:
    """Use VLM to reason about the image"""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=150
    )
    return {
        "success": True,
        "mode": "reason",
        "response": response.choices[0].message.content,
        "model": "SmolVLM-500M-Instruct"
    }


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/v1/commands")
async def commands():
    """Robot polls for commands here"""
    commands = list(command_queue)
    command_queue.clear()
    return {"commands": commands}


@app.get("/api/v1/frame")
async def get_latest_frame():
    """Browser polls for latest frame (as JSON with base64)"""
    if robot_state["latest_frame_base64"]:
        return {
            "success": True,
            "image": robot_state["latest_frame_base64"],
            "timestamp": robot_state["frame_timestamp"],
            "result": robot_state["latest_result"]
        }
    return {"success": False, "error": "No frame available"}


@app.get("/api/v1/frame.jpg")
async def get_frame_image():
    """Get latest frame as actual image (for img src)"""
    if robot_state["latest_frame"]:
        return Response(
            content=robot_state["latest_frame"],
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache"}
        )
    return Response(status_code=404)


@app.post("/api/v1/commands")
async def create_command(request: CommandRequest):
    """Browser sends commands for the robot"""
    cmd = {
        "command": request.command,
        "prompt": request.prompt,
        "timestamp": time.time()
    }
    command_queue.append(cmd)
    return {"success": True, "queued": cmd}


@app.post("/api/v1/actions")
async def analyze_image(
    file: UploadFile = File(...),
    mode: Literal["detect", "reason"] = Form("reason"),
    prompt: str | None = Form(None),
    target: str | None = Form(None),
):
    """
    Robot sends image here. Analyzes with YOLO or VLM, stores frame for browser.
    
    - file: Image from robot camera (required)
    - mode: "detect" for YOLO object detection, "reason" for VLM
    - prompt: Custom prompt for VLM (mode=reason)
    - target: Object to look for (mode=detect)
    """
    image_bytes = await file.read()
    robot_state["latest_frame"] = image_bytes
    robot_state["latest_frame_base64"] = base64.b64encode(image_bytes).decode('utf-8')
    robot_state["frame_timestamp"] = time.time()
    if mode == "detect":
        result = detect_objects(image_bytes, target=target)
    else:
        user_prompt = prompt or "Describe what you see in this image. What should the robot do?"
        result = reason_about_image(image_bytes, user_prompt)
    robot_state["latest_result"] = result
    robot_state["result_timestamp"] = time.time()
    commands = list(command_queue)
    command_queue.clear()
    result["commands"] = commands
    return JSONResponse(result)
