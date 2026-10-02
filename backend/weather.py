from datetime import datetime, timezone
from typing import Optional

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


async def fetch_weather(latitude: float, longitude: float) -> Optional[dict]:
    """Returns weather context for a location, or None if it cannot be fetched."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,precipitation,weather_code",
        "daily": "precipitation_sum",
        "past_days": 3,
        "forecast_days": 2,
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(OPEN_METEO_URL, params=params)
            response.raise_for_status()
            data = response.json()

        current = data["current"]
        daily = data["daily"]
        return {
            "source": "Open-Meteo",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "current": {
                "time": current["time"],
                "temperature_c": current["temperature_2m"],
                "precipitation_mm": current["precipitation"],
                "weather_code": current["weather_code"],
            },
            "daily_precipitation_mm": [
                {"date": d, "precipitation_mm": p}
                for d, p in zip(daily["time"], daily["precipitation_sum"])
            ],
        }
    except (httpx.HTTPError, KeyError, ValueError):
        return None