import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import (FastAPI, File, Form, HTTPException, Query, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

import annotator
import database
import video_processor
import video_store
import weather

app = FastAPI(title="CoastGuard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves saved images and videos at /media/...
annotator.ensure_dirs()
video_processor.ensure_dir()
app.mount("/media", StaticFiles(directory=annotator.MEDIA_DIR), name="media")

STATIC_DIR = Path(__file__).parent / "static"

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10
VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
}
MAX_VIDEO_MB = 100
MAX_LIVE_FRAME_BYTES = 2 * 1024 * 1024
MODEL_VERSION = "mock-v0"
IS_PLACEHOLDER = True


@app.on_event("startup")
def on_startup():
    database.init_db()
    database.init_weather_table()
    video_store.init_video_table()


@app.get("/api/health")
def health():
    return {"status": "ok"}


def run_detection(image: Image.Image) -> list[dict]:
    """PLACEHOLDER: returns fake detections.
    Replace this function with the real YOLO model later."""
    w, h = image.size
    return [
        {
            "class": "plastic_bottle",
            "confidence": 0.91,
            "bbox": [int(w * 0.10), int(h * 0.20), int(w * 0.30), int(h * 0.70)],
        },
        {
            "class": "plastic_bag",
            "confidence": 0.84,
            "bbox": [int(w * 0.50), int(h * 0.30), int(w * 0.75), int(h * 0.80)],
        },
    ]


def validate_location(latitude, longitude) -> str:
    """Returns the location accuracy label, or raises a 400 error."""
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=400, detail="Provide both latitude and longitude, or neither.")
    if latitude is None:
        return "missing"
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(status_code=400, detail="Latitude or longitude is out of range.")
    return "user_provided"


async def get_weather_context(obs_id, latitude, longitude):
    """Returns (weather_data, weather_status). Never raises."""
    if latitude is None:
        return None, "not_requested"
    data = await weather.fetch_weather(latitude, longitude)
    if data:
        database.save_weather(obs_id, data)
        return data, "available"
    return None, "unavailable"


@app.get("/api/weather")
async def get_weather(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
):
    data = await weather.fetch_weather(latitude, longitude)
    if data is None:
        raise HTTPException(status_code=503, detail="Weather service is unavailable right now.")
    return data


@app.post("/api/detection/image")
async def detect_image(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG or WebP images are allowed.")
    accuracy = validate_location(latitude, longitude)

    data = await file.read()
    if len(data) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_SIZE_MB} MB.")

    try:
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    detections = run_detection(image)

    obs_id = f"OBS-{uuid.uuid4().hex[:8].upper()}"
    created_at = database.save_observation(
        obs_id, latitude, longitude, accuracy,
        detections, MODEL_VERSION, IS_PLACEHOLDER,
    )
    images = annotator.save_images(obs_id, image, detections)
    weather_data, weather_status = await get_weather_context(obs_id, latitude, longitude)

    return {
        "observation_id": obs_id,
        "status": "completed",
        "media_type": "image",
        "created_at": created_at,
        "location": {"latitude": latitude, "longitude": longitude, "accuracy": accuracy},
        "detections": detections,
        "detection_count": len(detections),
        "images": images,
        "weather": weather_data,
        "weather_status": weather_status,
        "model_version": MODEL_VERSION,
        "is_placeholder": IS_PLACEHOLDER,
    }


@app.post("/api/detection/video")
async def detect_video(
    file: UploadFile = File(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    sample_every_seconds: float = Form(0.5),
    max_seconds: int = Form(60),
):
    if file.content_type not in VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Only MP4, MOV, AVI or WebM videos are allowed.")
    if not (0.1 <= sample_every_seconds <= 10):
        raise HTTPException(status_code=400, detail="sample_every_seconds must be between 0.1 and 10.")
    if not (1 <= max_seconds <= 120):
        raise HTTPException(status_code=400, detail="max_seconds must be between 1 and 120.")
    accuracy = validate_location(latitude, longitude)

    data = await file.read()
    if len(data) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Video is larger than {MAX_VIDEO_MB} MB.")

    obs_id = f"OBS-{uuid.uuid4().hex[:8].upper()}"
    try:
        summary, peak_image, peak_detections = await run_in_threadpool(
            video_processor.process_video,
            data,
            VIDEO_TYPES[file.content_type],
            obs_id,
            run_detection,
            sample_every_seconds,
            max_seconds,
        )
    except video_processor.VideoError as error:
        raise HTTPException(status_code=400, detail=str(error))

    # The observation record holds the detections of the busiest analysed frame.
    created_at = database.save_observation(
        obs_id, latitude, longitude, accuracy,
        peak_detections, MODEL_VERSION, IS_PLACEHOLDER,
    )
    images = annotator.save_images(obs_id, peak_image, peak_detections)
    video_store.save_video_summary(obs_id, summary)
    weather_data, weather_status = await get_weather_context(obs_id, latitude, longitude)

    return {
        "observation_id": obs_id,
        "status": "completed",
        "media_type": "video",
        "created_at": created_at,
        "location": {"latitude": latitude, "longitude": longitude, "accuracy": accuracy},
        "detections": peak_detections,
        "detection_count": len(peak_detections),
        "video": summary,
        "annotated_video_url": video_processor.video_url(obs_id),
        "images": images,
        "weather": weather_data,
        "weather_status": weather_status,
        "model_version": MODEL_VERSION,
        "is_placeholder": IS_PLACEHOLDER,
    }


@app.websocket("/api/detection/live")
async def live_detection(websocket: WebSocket):
    """The client sends one JPEG frame (binary) and gets the detections back as JSON."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) > MAX_LIVE_FRAME_BYTES:
                await websocket.send_json({"error": "Frame is too large."})
                continue
            try:
                image = Image.open(BytesIO(data)).convert("RGB")
            except (UnidentifiedImageError, OSError):
                await websocket.send_json({"error": "Frame is not a valid image."})
                continue

            started = time.perf_counter()
            detections = await run_in_threadpool(run_detection, image)
            await websocket.send_json({
                "width": image.width,
                "height": image.height,
                "detections": detections,
                "processing_ms": round((time.perf_counter() - started) * 1000),
                "model_version": MODEL_VERSION,
                "is_placeholder": IS_PLACEHOLDER,
            })
    except WebSocketDisconnect:
        pass


@app.get("/live-test", response_class=HTMLResponse)
def live_test_page():
    """A small test page for the live camera (reference for the frontend)."""
    return (STATIC_DIR / "live_test.html").read_text(encoding="utf-8")


@app.get("/api/observations/{obs_id}")
def get_observation(obs_id: str):
    obs = database.get_observation(obs_id)
    if obs is None:
        raise HTTPException(status_code=404, detail="Observation not found.")

    weather_data = database.get_weather(obs_id)
    obs["weather"] = weather_data
    if weather_data:
        obs["weather_status"] = "available"
    elif obs["location"]["latitude"] is None:
        obs["weather_status"] = "not_requested"
    else:
        obs["weather_status"] = "unavailable"

    video = video_store.get_video_summary(obs_id)
    obs["media_type"] = "video" if video else "image"
    obs["video"] = video
    obs["annotated_video_url"] = video_processor.video_url(obs_id)
    obs["images"] = annotator.image_urls(obs_id)
    return obs


@app.get("/api/locations")
def get_locations():
    return {"locations": database.list_locations()}


@app.get("/api/analytics")
def get_analytics():
    return database.get_analytics()