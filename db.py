import sqlite3
from typing import Any

DB_PATH = "soccer.db"


def init_db() -> None:
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                competition TEXT NOT NULL,
                season INTEGER NOT NULL,
                utc_date TEXT,
                status TEXT,
                matchday INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_comp_season
            ON matches (competition, season)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matches_comp_season_date
            ON matches (competition, season, utc_date)
            """
        )


def save_matches(matches: list[dict[str, Any]], competition: str, season: int) -> None:
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute(
            "DELETE FROM matches WHERE competition = ? AND season = ?",
            (competition, season),
        )

        conn.executemany(
            """
            INSERT OR REPLACE INTO matches (
                match_id,
                competition,
                season,
                utc_date,
                status,
                matchday,
                home_team,
                away_team,
                home_score,
                away_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    match.get("match_id"),
                    competition,
                    season,
                    match.get("utc_date"),
                    match.get("status"),
                    match.get("matchday"),
                    match.get("home_team"),
                    match.get("away_team"),
                    match.get("home_score"),
                    match.get("away_score"),
                )
                for match in matches
            ],
        )


def get_matches(competition: str, season: int) -> list[dict[str, Any]]:
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                match_id,
                competition,
                season,
                utc_date,
                status,
                matchday,
                home_team,
                away_team,
                home_score,
                away_score
            FROM matches
            WHERE competition = ? AND season = ?
            ORDER BY utc_date DESC
            """,
            (competition, season),
        ).fetchall()

    return [dict(row) for row in rows]


def get_team_matches(competition: str, season: int, team_name: str) -> list[dict[str, Any]]:
    """Return matches for a single team (home or away) within competition and season."""
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                match_id,
                competition,
                season,
                utc_date,
                status,
                matchday,
                home_team,
                away_team,
                home_score,
                away_score
            FROM matches
            WHERE competition = ?
              AND season = ?
              AND (home_team = ? OR away_team = ?)
            ORDER BY utc_date ASC
            """,
            (competition, season, team_name, team_name),
        ).fetchall()

    return [dict(row) for row in rows]
