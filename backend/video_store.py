import json

from database import get_conn


def init_video_table():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS observation_video (
                observation_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL
            )
            """
        )


def save_video_summary(obs_id, summary):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO observation_video (observation_id, summary) VALUES (?, ?)",
            (obs_id, json.dumps(summary)),
        )


def get_video_summary(obs_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT summary FROM observation_video WHERE observation_id = ?", (obs_id,)
        ).fetchone()
    return json.loads(row["summary"]) if row else None