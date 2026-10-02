import uuid
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

import annotator
import database
import weather

app = FastAPI(title="CoastGuard AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves saved images at /media/original/... and /media/annotated/...
annotator.ensure_dirs()
app.mount("/media", StaticFiles(directory=annotator.MEDIA_DIR), name="media")

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10
MODEL_VERSION = "mock-v0"
IS_PLACEHOLDER = True


@app.on_event("startup")
def on_startup():
    database.init_db()
    database.init_weather_table()


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
    # 1. Check the file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG or WebP images are allowed.")

    # 2. Check the location (both values or neither)
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=400, detail="Provide both latitude and longitude, or neither.")
    if latitude is not None and not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(status_code=400, detail="Latitude or longitude is out of range.")
    accuracy = "user_provided" if latitude is not None else "missing"

    # 3. Check the file size
    data = await file.read()
    if len(data) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image is larger than {MAX_SIZE_MB} MB.")

    # 4. Open the image, fix rotation, and make sure it is a real image
    try:
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    # 5. Run detection
    detections = run_detection(image)

    # 6. Save the observation and its images
    obs_id = f"OBS-{uuid.uuid4().hex[:8].upper()}"
    created_at = database.save_observation(
        obs_id, latitude, longitude, accuracy,
        detections, MODEL_VERSION, IS_PLACEHOLDER,
    )
    images = annotator.save_images(obs_id, image, detections)

    # 7. Weather context (optional: detection still succeeds if it fails)
    weather_data = None
    weather_status = "not_requested"
    if latitude is not None:
        weather_data = await weather.fetch_weather(latitude, longitude)
        if weather_data:
            weather_status = "available"
            database.save_weather(obs_id, weather_data)
        else:
            weather_status = "unavailable"

    # 8. Return the response
    return {
        "observation_id": obs_id,
        "status": "completed",
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

    obs["images"] = annotator.image_urls(obs_id)
    return obs


@app.get("/api/locations")
def get_locations():
    return {"locations": database.list_locations()}


@app.get("/api/analytics")
def get_analytics():
    return database.get_analytics()