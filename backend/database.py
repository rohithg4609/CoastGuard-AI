import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "coastguard.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the table if it does not exist yet."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                location_accuracy TEXT NOT NULL,
                detections TEXT NOT NULL,
                detection_count INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                is_placeholder INTEGER NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )


def init_weather_table():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observation_weather (
                observation_id TEXT PRIMARY KEY,
                weather TEXT NOT NULL
            )
            """
        )


def _row_to_dict(row):
    return {
        "observation_id": row["id"],
        "created_at": row["created_at"],
        "location": {
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "accuracy": row["location_accuracy"],
        },
        "detections": json.loads(row["detections"]),
        "detection_count": row["detection_count"],
        "model_version": row["model_version"],
        "is_placeholder": bool(row["is_placeholder"]),
        "review_status": row["review_status"],
    }


def save_observation(obs_id, latitude, longitude, accuracy,
                     detections, model_version, is_placeholder):
    created_at = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO observations
            (id, created_at, latitude, longitude, location_accuracy,
             detections, detection_count, model_version, is_placeholder)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (obs_id, created_at, latitude, longitude, accuracy,
             json.dumps(detections), len(detections), model_version,
             int(is_placeholder)),
        )
    return created_at


def get_observation(obs_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM observations WHERE id = ?", (obs_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_locations():
    """Observations that have coordinates, for the map."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, latitude, longitude,
                   location_accuracy, detection_count
            FROM observations
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY created_at DESC
            """
        ).fetchall()
    return [
        {
            "observation_id": r["id"],
            "created_at": r["created_at"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "accuracy": r["location_accuracy"],
            "detection_count": r["detection_count"],
        }
        for r in rows
    ]


def save_weather(obs_id, weather):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO observation_weather (observation_id, weather) VALUES (?, ?)",
            (obs_id, json.dumps(weather)),
        )


def get_weather(obs_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT weather FROM observation_weather WHERE observation_id = ?", (obs_id,)
        ).fetchone()
    return json.loads(row["weather"]) if row else None


def get_analytics():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT created_at, latitude, longitude, detections, is_placeholder FROM observations"
        ).fetchall()

    per_day = {}
    classes = {}
    sites = {}
    located = 0
    placeholder = 0

    for r in rows:
        day = r["created_at"][:10]
        per_day[day] = per_day.get(day, 0) + 1
        if r["is_placeholder"]:
            placeholder += 1
        for d in json.loads(r["detections"]):
            classes[d["class"]] = classes.get(d["class"], 0) + 1
        if r["latitude"] is not None and r["longitude"] is not None:
            located += 1
            # rounding to 3 decimals groups observations within roughly 100 m
            key = (round(r["latitude"], 3), round(r["longitude"], 3))
            sites[key] = sites.get(key, 0) + 1

    return {
        "total_observations": len(rows),
        "observations_with_location": located,
        "placeholder_observations": placeholder,
        "observations_per_day": [
            {"date": d, "count": c} for d, c in sorted(per_day.items())
        ],
        "detections_by_class": [
            {"class": k, "count": v}
            for k, v in sorted(classes.items(), key=lambda x: -x[1])
        ],
        "observations_by_site": [
            {"latitude": lat, "longitude": lon, "observation_count": c}
            for (lat, lon), c in sorted(sites.items(), key=lambda x: -x[1])
        ],
        "sites_with_repeat_observations": sum(1 for c in sites.values() if c >= 2),
        "note": "Counts describe recorded detections in processed images, not the total waste at any site.",
    }