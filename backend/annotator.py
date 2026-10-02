from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MEDIA_DIR = Path(__file__).parent / "media"
ORIGINAL_DIR = MEDIA_DIR / "original"
ANNOTATED_DIR = MEDIA_DIR / "annotated"


def ensure_dirs():
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    """Returns a copy of the image with a box and label for each detection."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(2, image.width // 250)
    font = ImageFont.load_default(size=max(12, image.width // 40))

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)

        label = f'{d["class"]} {d["confidence"]:.2f}'
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 6], fill=(255, 0, 0))
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255), font=font)

    return annotated


def image_urls(obs_id: str) -> dict:
    """URLs of the saved images, or None if a file does not exist."""
    original = ORIGINAL_DIR / f"{obs_id}.jpg"
    annotated = ANNOTATED_DIR / f"{obs_id}.jpg"
    return {
        "original_url": f"/media/original/{obs_id}.jpg" if original.exists() else None,
        "annotated_url": f"/media/annotated/{obs_id}.jpg" if annotated.exists() else None,
    }


def save_images(obs_id: str, image: Image.Image, detections: list) -> dict:
    ensure_dirs()
    image.save(ORIGINAL_DIR / f"{obs_id}.jpg", "JPEG", quality=90)
    draw_boxes(image, detections).save(ANNOTATED_DIR / f"{obs_id}.jpg", "JPEG", quality=90)
    return image_urls(obs_id)