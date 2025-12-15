from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pathlib import Path
from typing import Literal
import json
import re
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
yolo_model: YOLO = YOLO("yolov8n.pt")

# Global state for identified targets from VLM
identified_targets: list[str] = []

SYSTEM_PROMPT = """You are a vision system for a robot car. Analyze the image and provide a brief description of what you see.
Focus on objects, people, obstacles, and any relevant environmental details that would help the robot navigate or interact with its surroundings."""

IDENTIFY_SYSTEM_PROMPT = """You are a vision system for a robot. Your job is to identify objects in the scene that can be detected by a YOLO model.

YOLO can detect these classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

Respond ONLY with a JSON object in this exact format:
{"targets": ["object1", "object2", ...]}

Only include objects that are actually visible in the image AND are in the YOLO class list above."""

IDENTIFY_PROMPT = "What objects are in front of me? Identify all visible objects that match YOLO detection classes. Return ONLY valid JSON."

def detect_objects(image_bytes: bytes, target: str | None = None, conf_threshold: float = 0.5) -> dict:
    """Run YOLO object detection on image using Ultralytics YOLOv8
    
    Args:
        target: Single target or multiple targets separated by |
                e.g., "person" or "person|cup|bottle"
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"success": False, "error": "Failed to decode image"}
    results = yolo_model(img, conf=conf_threshold, verbose=False)
    detections = []
    target_found = False
    target_list = [t.strip().lower() for t in target.split("|")] if target else []
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
            is_target_match = False
            if target_list:
                for t in target_list:
                    if t in class_name.lower():
                        is_target_match = True
                        target_found = True
                        break
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "box": bbox,  # [x, y, w, h]
                "xyxy": [int(x) for x in xyxy],  # [x1, y1, x2, y2]
                "is_target": is_target_match if target_list else None
            })
    result = {
        "success": True,
        "mode": "detect",
        "detections": detections,
        "count": len(detections)
    }
    if target:
        result["target"] = target
        result["target_list"] = target_list
        result["target_found"] = target_found
        # Filter to only target detections for convenience
        result["target_detections"] = [d for d in detections if d.get("is_target")]
    return result


def reason_about_image(image_bytes: bytes, system_prompt: str, prompt: str) -> dict:
    """Use VLM to reason about the image"""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": system_prompt
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


def parse_targets_from_response(response_text: str) -> list[str]:
    """Extract target list from VLM JSON response"""
    try:
        data = json.loads(response_text)
        if isinstance(data, dict) and "targets" in data:
            return [t.lower().strip() for t in data["targets"] if isinstance(t, str)]
    except json.JSONDecodeError:
        pass
    json_match = re.search(r'\{[^{}]*"targets"\s*:\s*\[[^\]]*\][^{}]*\}', response_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "targets" in data:
                return [t.lower().strip() for t in data["targets"] if isinstance(t, str)]
        except json.JSONDecodeError:
            pass
    return []


def identify_targets(image_bytes: bytes, system_prompt: str, prompt: str) -> dict:
    """Use VLM to identify objects and generate YOLO targets"""
    global identified_targets
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": system_prompt
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
        max_tokens=200
    )
    response_text = response.choices[0].message.content
    targets = parse_targets_from_response(response_text)
    if targets:
        identified_targets = targets
    return {
        "success": True,
        "mode": "identify",
        "response": response_text,
        "targets": targets,
        "targets_saved": len(targets) > 0,
        "model": "SmolVLM-500M-Instruct"
    }


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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


@app.post("/api/v1/actions")
async def analyze_image(
    file: UploadFile = File(...),
    mode: Literal["detect", "identify"] = Form("detect"),
    system_prompt: str = Form(IDENTIFY_SYSTEM_PROMPT),
    prompt: str = Form(IDENTIFY_PROMPT),
    target: str = Form(""),
):
    """
    Robot sends image here. Analyzes with YOLO or VLM, stores frame for browser.
    
    - file: Image from robot camera (required)
    - mode: "detect" for YOLO object detection, "reason" for VLM, "identify" to find targets for YOLO
    - system_prompt: Custom system prompt for VLM (mode=reason)
    - prompt: Custom prompt for VLM (mode=reason)
    - target: Object to look for (mode=detect). If not provided, uses targets from last "identify" call
    """
    image_bytes = await file.read()
    robot_state["latest_frame"] = image_bytes
    robot_state["latest_frame_base64"] = base64.b64encode(image_bytes).decode('utf-8')
    robot_state["frame_timestamp"] = time.time()
    if mode == "identify":
        result = identify_targets(image_bytes, system_prompt, prompt)
    elif mode == "detect":
        if target:
            identified_targets.clear()
            identified_targets.extend([t.strip().lower() for t in target.split("|") if t.strip()])
            result = detect_objects(image_bytes, target=target)
        else:
            result = detect_objects(image_bytes, target="|".join(identified_targets) if identified_targets else "")
        result["identified_targets"] = identified_targets
    robot_state["latest_result"] = result
    robot_state["result_timestamp"] = time.time()
    return JSONResponse(result)
