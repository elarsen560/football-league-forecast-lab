import json
import os
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://api.football-data.org/v4"


def _normalize_status(status: object, full_time: dict) -> object:
    """Normalize provider date-string placeholders for scoreless scheduled fixtures."""
    if (
        isinstance(status, str)
        and full_time.get("home") is None
        and full_time.get("away") is None
    ):
        try:
            datetime.fromisoformat(status.replace("Z", "+00:00"))
        except ValueError:
            pass
        else:
            return "SCHEDULED"
    return status


def fetch_matches(competition: str = "DED", season: int = 2025) -> list[dict]:
    api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FOOTBALL_DATA_API_KEY environment variable")

    query = urlencode({"season": season})
    url = f"{BASE_URL}/competitions/{competition}/matches?{query}"
    request = Request(url, headers={"X-Auth-Token": api_key})

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"football-data API error ({exc.code}): {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"football-data request failed: {exc.reason}") from exc

    matches = payload.get("matches", [])
    normalized = []
    for match in matches:
        full_time = match.get("score", {}).get("fullTime", {})
        normalized.append(
            {
                "match_id": match.get("id"),
                "competition": competition,
                "season": season,
                "utc_date": match.get("utcDate"),
                "status": _normalize_status(match.get("status"), full_time),
                "matchday": match.get("matchday"),
                "home_team": match.get("homeTeam", {}).get("name"),
                "away_team": match.get("awayTeam", {}).get("name"),
                "home_score": full_time.get("home"),
                "away_score": full_time.get("away"),
            }
        )

    return normalized
