from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import JSONResponse
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
    # Navigation state for gap detection
    "initial_objects": [],  # List of class names when navigation started
    "is_navigating": False,
    "gap_left_class": None,  # Class name of object on left side of gap
    "gap_right_class": None,  # Class name of object on right side of gap
}
yolo_model: YOLO = YOLO("yolo11s.pt")

# Global state for identified targets from VLM
identified_targets: list[str] = []

# Threshold for considering robot "centered" on the gap (in pixels from center)
CENTER_THRESHOLD = 50  # Adjust based on camera resolution and desired precision

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


def find_largest_gap(detections: list[dict]) -> dict | None:
    """
    Find the largest gap between detected objects.
    
    Returns a dict with:
    - left_object: detection on the left side of gap
    - right_object: detection on the right side of gap
    - gap_center_x: x-coordinate of the gap center
    - gap_width: width of the gap
    """
    if len(detections) < 2:
        return None
    # Sort detections by their center x-coordinate
    sorted_detections = sorted(detections, key=lambda d: d["xyxy"][0] + (d["xyxy"][2] - d["xyxy"][0]) / 2)
    # Find the largest gap between consecutive objects
    largest_gap = None
    largest_gap_width = 0
    for i in range(len(sorted_detections) - 1):
        left_obj = sorted_detections[i]
        right_obj = sorted_detections[i + 1]
        # Gap is from right edge of left object to left edge of right object
        left_edge = left_obj["xyxy"][2]  # x2 of left object
        right_edge = right_obj["xyxy"][0]  # x1 of right object
        gap_width = right_edge - left_edge
        if gap_width > largest_gap_width:
            largest_gap_width = gap_width
            largest_gap = {
                "left_object": left_obj,
                "right_object": right_obj,
                "gap_center_x": (left_edge + right_edge) / 2,
                "gap_width": gap_width,
            }
    return largest_gap


def calculate_navigation_action(gap_info: dict, image_width: int = 640) -> str:
    """
    Calculate the navigation action based on gap position relative to image center.
    
    Returns: "drive", "turn_left", or "turn_right"
    """
    image_center_x = image_width / 2
    gap_center_x = gap_info["gap_center_x"]
    offset = gap_center_x - image_center_x
    if abs(offset) <= CENTER_THRESHOLD:
        # Centered on the gap, drive forward
        return "drive"
    elif offset > 0:
        # Gap is to the right of center, robot view is tilted left
        # Need to turn right to center the gap
        return "turn_right"
    else:
        # Gap is to the left of center, robot view is tilted right
        # Need to turn left to center the gap
        return "turn_left"


def check_initial_objects_present(current_detections: list[dict], initial_classes: list[str]) -> bool:
    """
    Check if any of the initial objects are still present in current detections.
    """
    current_classes = {d["class"].lower() for d in current_detections}
    for initial_class in initial_classes:
        if initial_class.lower() in current_classes:
            return True
    return False


def identify_targets(image_bytes: bytes, system_prompt: str, prompt: str) -> dict:
    """Use VLM to identify objects and generate YOLO targets"""
    global identified_targets
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    response = client.chat.completions.create(
        model="SmolVLM-500M-Instruct",
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
    Returns an action for the robot to execute.
    
    Actions:
    - "stop": Default action, also returned in identify mode or on errors
    - "drive": Robot is centered on gap, drive forward
    - "turn_left": Robot view is tilted right, turn left to center
    - "turn_right": Robot view is tilted left, turn right to center
    - "goal": Initial objects no longer detected, navigation complete
    
    - file: Image from robot camera (required)
    - mode: "detect" for YOLO object detection with navigation, "identify" to find targets for YOLO
    - system_prompt: Custom system prompt for VLM (mode=identify)
    - prompt: Custom prompt for VLM (mode=identify)
    - target: Object to look for (mode=detect). If not provided, uses targets from last "identify" call
    """
    action = "stop"
    error_message = "Unknown error"
    try:
        image_bytes = await file.read()
        robot_state["latest_frame"] = image_bytes
        robot_state["latest_frame_base64"] = base64.b64encode(image_bytes).decode('utf-8')
        robot_state["frame_timestamp"] = time.time()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image_width = img.shape[1] if img is not None else 640
        if mode == "identify":
            # Identify mode: detect objects with VLM and reset navigation state
            result = identify_targets(image_bytes, system_prompt, prompt)
            # Reset navigation state when identifying new targets
            robot_state["initial_objects"] = []
            robot_state["is_navigating"] = False
            robot_state["gap_left_class"] = None
            robot_state["gap_right_class"] = None
            action = "stop"  # Always stop in identify mode
        elif mode == "detect":
            if target:
                identified_targets.clear()
                identified_targets.extend([t.strip().lower() for t in target.split("|") if t.strip()])
                result = detect_objects(image_bytes, target=target)
            else:
                result = detect_objects(image_bytes, target="|".join(identified_targets) if identified_targets else "")
            result["identified_targets"] = list(identified_targets)
            detections = result.get("detections", [])
            # If we're navigating, check if initial objects are still present
            if robot_state["is_navigating"]:
                initial_classes = robot_state["initial_objects"]
                if not check_initial_objects_present(detections, initial_classes):
                    # Initial objects no longer detected - goal reached!
                    action = "goal"
                    result["navigation_status"] = "goal_reached"
                    result["reason"] = "Initial objects no longer in view"
                    # Reset navigation state
                    robot_state["is_navigating"] = False
                else:
                    # Still navigating - calculate action based on gap
                    gap_info = find_largest_gap(detections, image_width)
                    if gap_info:
                        action = calculate_navigation_action(gap_info, image_width)
                        result["gap_info"] = {
                            "left_object": gap_info["left_object"]["class"],
                            "right_object": gap_info["right_object"]["class"],
                            "gap_center_x": gap_info["gap_center_x"],
                            "gap_width": gap_info["gap_width"],
                            "image_center_x": image_width / 2,
                        }
                        result["navigation_status"] = "navigating"
                    else:
                        # Not enough objects to find a gap
                        action = "stop"
                        result["navigation_status"] = "no_gap_found"
                        result["reason"] = "Need at least 2 objects to find a gap"
            else:
                # Not navigating yet - start navigation if we detect objects
                if len(detections) >= 2:
                    # Start navigation
                    gap_info = find_largest_gap(detections, image_width)
                    if gap_info:
                        # Store initial objects for goal detection
                        robot_state["initial_objects"] = list(set(d["class"].lower() for d in detections))
                        robot_state["is_navigating"] = True
                        robot_state["gap_left_class"] = gap_info["left_object"]["class"]
                        robot_state["gap_right_class"] = gap_info["right_object"]["class"]
                        action = calculate_navigation_action(gap_info, image_width)
                        result["gap_info"] = {
                            "left_object": gap_info["left_object"]["class"],
                            "right_object": gap_info["right_object"]["class"],
                            "gap_center_x": gap_info["gap_center_x"],
                            "gap_width": gap_info["gap_width"],
                            "image_center_x": image_width / 2,
                        }
                        result["navigation_status"] = "started"
                    else:
                        action = "stop"
                        result["navigation_status"] = "no_gap_found"
                else:
                    action = "stop"
                    result["navigation_status"] = "waiting"
                    result["reason"] = f"Need at least 2 objects, found {len(detections)}"
        result["action"] = action
        result["navigation_state"] = {
            "is_navigating": robot_state["is_navigating"],
            "initial_objects": robot_state["initial_objects"],
            "gap_left_class": robot_state["gap_left_class"],
            "gap_right_class": robot_state["gap_right_class"],
        }
        robot_state["latest_result"] = result
        robot_state["result_timestamp"] = time.time()
        return JSONResponse(result)
    except Exception as e:
        error_message = str(e)
        result = {
            "success": False,
            "error": error_message,
            "action": "stop",
            "mode": mode,
        }
        robot_state["latest_result"] = result
        robot_state["result_timestamp"] = time.time()
        return JSONResponse(result)
