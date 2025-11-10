from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pathlib import Path
import base64

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
SYSTEM_PROMPT = """You are a vision system for a robot. Analyze the image and provide a brief description of what you see.
Focus on objects, people, obstacles, and any relevant environmental details that would help the robot navigate or interact with its surroundings."""
USER_PROMPT = "Describe what you see in this image in one sentence."

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/v1/actions")
async def analyze_image(file: UploadFile = File(...)):
    """Receive an image and return VLM inference results"""
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode('utf-8')
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
                        "text": USER_PROMPT
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
        max_tokens=100
    )
    caption = response.choices[0].message.content
    return JSONResponse({
        "success": True,
        "caption": caption,
        "model": "SmolVLM-500M-Instruct"
    })