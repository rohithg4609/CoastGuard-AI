import os
import tempfile
from typing import Callable, Optional

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image

import annotator

VIDEO_DIR = annotator.MEDIA_DIR / "annotated_video"
MAX_FRAME_SIZE = 960


class VideoError(Exception):
    """Raised when a video cannot be read or written."""


def ensure_dir():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def video_url(obs_id: str) -> Optional[str]:
    path = VIDEO_DIR / f"{obs_id}.mp4"
    return f"/media/annotated_video/{obs_id}.mp4" if path.exists() else None


def process_video(video_bytes: bytes, suffix: str, obs_id: str, detect_fn: Callable,
                  detect_every_seconds: float = 0.5, max_seconds: int = 60):
    """Reads the whole video, runs detection every few frames, draws the boxes
    on every frame and writes a new .mp4.
    Returns (summary, peak_image, peak_detections)."""
    ensure_dir()
    out_path = VIDEO_DIR / f"{obs_id}.mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    cap = None
    writer = None
    success = False
    try:
        tmp.write(video_bytes)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise VideoError("Could not open the video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0 or fps > 240:
            fps = 30.0
        max_frames = int(max_seconds * fps)
        detect_step = max(1, int(round(fps * detect_every_seconds)))

        out_w = out_h = 0
        current = []        # detections currently drawn on the video
        analysed = 0
        written = 0
        classes = {}
        peak = None         # (count, image, detections, frame_index, timestamp)

        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            # Set the output size from the first frame (handles rotated phone videos)
            if writer is None:
                h, w = frame.shape[:2]
                scale = min(1.0, MAX_FRAME_SIZE / max(w, h))
                out_w = max(2, int(w * scale) // 2 * 2)
                out_h = max(2, int(h * scale) // 2 * 2)
                writer = imageio_ffmpeg.write_frames(
                    str(out_path), (out_w, out_h), fps=fps, codec="libx264",
                    pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2,
                    output_params=["-movflags", "+faststart"],
                    ffmpeg_log_level="error",
                )
                writer.send(None)

            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))

            # OpenCV gives BGR colours; models and Pillow expect RGB
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if written % detect_step == 0:
                current = detect_fn(image)
                analysed += 1
                timestamp = round(written / fps, 2)
                if peak is None or len(current) > peak[0]:
                    peak = (len(current), image, current, written, timestamp)
                per_frame = {}
                for d in current:
                    per_frame[d["class"]] = per_frame.get(d["class"], 0) + 1
                for name, count in per_frame.items():
                    entry = classes.setdefault(
                        name, {"class": name, "max_in_one_frame": 0, "frames_seen": 0}
                    )
                    entry["max_in_one_frame"] = max(entry["max_in_one_frame"], count)
                    entry["frames_seen"] += 1

            annotated = annotator.draw_boxes(image, current)
            writer.send(np.asarray(annotated).tobytes())
            written += 1

        if writer is None or analysed == 0:
            raise VideoError("Could not read any frames from the video.")

        truncated = written >= max_frames and cap.read()[0]
        writer.close()
        writer = None
        success = True

        summary = {
            "duration_s": round(written / fps, 1),
            "fps": round(fps, 1),
            "frames_written": written,
            "frames_analyzed": analysed,
            "detect_every_seconds": detect_every_seconds,
            "was_truncated": bool(truncated),
            "peak_frame": {
                "frame_index": peak[3],
                "timestamp_s": peak[4],
                "detection_count": peak[0],
            },
            "classes": sorted(classes.values(), key=lambda c: -c["max_in_one_frame"]),
            "note": "Counts are per analysed frame, not totals of waste in the video.",
        }
        return summary, peak[1], peak[2]

    except VideoError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise VideoError(f"Could not process the video: {error}")
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        if not success and out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        try:
            os.unlink(tmp.name)
        except OSError:
            pass