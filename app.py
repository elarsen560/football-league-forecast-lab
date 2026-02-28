import csv
import math
import os
import random
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st

from data_client import fetch_matches
from db import get_matches, init_db, save_matches
from elo import DEFAULT_ELO, DEFAULT_HOME_ADVANTAGE, compute_elo_ratings, predict_match

init_db()

FINISHED_STATUS = "FINISHED"
UPCOMING_STATUSES = {"SCHEDULED", "TIMED"}
MISSING_STARTING_ELO_MSG = (
    f"starting_elo.csv not found. Using default starting Elo ratings ({int(DEFAULT_ELO)})."
)
INVALID_STARTING_ELO_MSG = (
    f"Failed to parse starting_elo.csv. Using default starting Elo ratings ({int(DEFAULT_ELO)})."
)
MISSING_MODEL_CONFIG_MSG = "model_config.csv not found. Using default home advantage."
INVALID_MODEL_CONFIG_MSG = "Invalid model_config.csv. Using default home advantage."
NO_UPCOMING_MSG = "No upcoming matches available for probability prediction."
NO_COMPLETED_MSG = "No FINISHED matches available."
NO_RATINGS_MSG = "No FINISHED matches available to compute Elo ratings."
NO_STANDINGS_MSG = "No completed matches available to compute standings."
TABLE_ROW_HEIGHT = 35
TABLE_EXTRA_HEIGHT = 20
PROBABILITY_COLUMNS = ("p_win", "p_draw", "p_loss")
COMPETITION_OPTIONS = {
    "Eredivisie": "DED",
    "Premier League": "PL",
    "Bundesliga": "BL1",
    "Serie A": "SA",
    "La Liga": "PD",
}
SEASON_OPTIONS = [2025]
MONTE_CARLO_MIN = 100
MONTE_CARLO_MAX = 20000
MONTE_CARLO_DEFAULT = 10000
MONTE_CARLO_STEP = 100
CALIBRATION_BIN_START = 0.0
CALIBRATION_BIN_END = 0.35
CALIBRATION_BIN_WIDTH = 0.05
CALIBRATION_FULL_BIN_START = 0.0
CALIBRATION_FULL_BIN_END = 1.0
CALIBRATION_FULL_BIN_WIDTH = 0.1

FOOTER_LINES = [
    "Model v0.2.0",
    "Last updated 2026/02/27",
    "K = 20",
    "Home Advantage = League Specific (default 75)",
    "Goal-Difference Multiplier = ON",
    "Dynamic Draw Model = ON",
    "Data source: football-data.org",
]


def format_display_date(utc_date: str) -> str:
    if not utc_date:
        return ""
    try:
        return datetime.fromisoformat(utc_date.replace("Z", "+00:00")).strftime("%b %d, %Y")
    except ValueError:
        return utc_date


def clean_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop common index-like columns and return a clean, reindexed DataFrame."""
    df = df.drop(
        columns=[col for col in df.columns if col == "index" or str(col).startswith("Unnamed")],
        errors="ignore",
    )
    return df.reset_index(drop=True)


def full_table_height(df: pd.DataFrame) -> int:
    """Return a consistent full-table height so all rows render without vertical scrolling."""
    return TABLE_ROW_HEIGHT * (len(df) + 1) + TABLE_EXTRA_HEIGHT


def _file_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


@st.cache_data(show_spinner=False)
def _load_home_advantage_config_cached(path: str, mtime: float | None) -> tuple[dict[str, float], str | None]:
    del mtime
    ha_map: dict[str, float] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None:
                raise ValueError("model_config.csv is empty")
            required_columns = {"competition", "home_advantage"}
            if not required_columns.issubset(set(reader.fieldnames)):
                raise ValueError("model_config.csv must include competition and home_advantage columns")
            for row in reader:
                competition = (row.get("competition") or "").strip()
                home_advantage_raw = (row.get("home_advantage") or "").strip()
                if competition == "" and home_advantage_raw == "":
                    continue
                if competition == "" or home_advantage_raw == "":
                    raise ValueError("model_config.csv contains incomplete rows")
                if competition in ha_map:
                    raise ValueError("model_config.csv contains duplicate competition values")
                ha_value = float(home_advantage_raw)
                if ha_value < 0.0:
                    raise ValueError("home_advantage must be non-negative")
                ha_map[competition] = ha_value
        return ha_map, None
    except FileNotFoundError:
        return {}, MISSING_MODEL_CONFIG_MSG
    except Exception:
        return {}, INVALID_MODEL_CONFIG_MSG


def load_home_advantage_config(path: str = "model_config.csv") -> tuple[dict[str, float], str | None]:
    """Load home-advantage settings by competition with mtime-sensitive cache invalidation."""
    return _load_home_advantage_config_cached(path, _file_mtime(path))


def build_calibration_df(
    matches_df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    bin_edges: list[float],
) -> pd.DataFrame:
    """Build binned calibration data with bin-center x and empirical outcome rate y."""
    rows = []
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        is_last = i == (len(bin_edges) - 2)
        if is_last:
            mask = (matches_df[pred_col] >= bin_start) & (matches_df[pred_col] <= bin_end)
            bin_label = f"[{bin_start:.2f}, {bin_end:.2f}]"
        else:
            mask = (matches_df[pred_col] >= bin_start) & (matches_df[pred_col] < bin_end)
            bin_label = f"[{bin_start:.2f}, {bin_end:.2f})"
        subset = matches_df.loc[mask]
        if subset.empty:
            continue
        n = int(len(subset))
        k = int(subset[actual_col].sum())
        actual_rate = float(k / n)
        ci_low, ci_high = wilson_ci(k, n, z=1.96)
        rows.append(
            {
                "bin_start": bin_start,
                "bin_end": bin_end,
                "bin_center": (bin_start + bin_end) / 2.0,
                "actual_rate": actual_rate,
                "k": k,
                "n": n,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "bin_label": bin_label,
            }
        )
    return pd.DataFrame(rows)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (p_hat + (z2 / (2.0 * n))) / denom
    half_width = (z * ((p_hat * (1.0 - p_hat) + (z2 / (4.0 * n))) / n) ** 0.5) / denom
    ci_low = max(0.0, center - half_width)
    ci_high = min(1.0, center + half_width)
    return ci_low, ci_high


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_matches_cached(competition: str, season: int) -> tuple[list[dict], str]:
    """Fetch matches from API with a shared 1-hour cache and return fetch timestamp (UTC)."""
    matches = fetch_matches(competition=competition, season=season)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()
    return matches, fetched_at_utc


def load_starting_ratings_csv(
    competition: str,
    path: str = "starting_elo.csv",
) -> tuple[dict[str, float], str | None]:
    """Load competition-filtered seed Elo ratings from CSV."""
    starting_ratings: dict[str, float] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row_competition = row.get("competition")
                team = row.get("team")
                rating = row.get("rating")
                if row_competition is None or team is None or rating is None:
                    raise ValueError("CSV must include competition, team and rating columns")
                if row_competition != competition:
                    continue
                starting_ratings[team] = float(rating)
        return starting_ratings, None
    except FileNotFoundError:
        return {}, MISSING_STARTING_ELO_MSG
    except Exception:
        return {}, INVALID_STARTING_ELO_MSG


def split_matches_by_status(stored_matches: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Split matches into finished/completed/upcoming sets with app-specific ordering."""
    finished_matches = [m for m in stored_matches if m.get("status") == FINISHED_STATUS]
    finished_matches = sorted(finished_matches, key=lambda m: m.get("utc_date") or "")

    completed_matches = [m for m in stored_matches if m.get("status") == FINISHED_STATUS]
    completed_matches = sorted(completed_matches, key=lambda m: m.get("utc_date") or "", reverse=True)

    upcoming_matches = [m for m in stored_matches if m.get("status") in UPCOMING_STATUSES]
    upcoming_matches = sorted(
        upcoming_matches,
        key=lambda m: (m.get("matchday") if m.get("matchday") is not None else 9999, m.get("utc_date") or ""),
    )
    return finished_matches, completed_matches, upcoming_matches


def compute_league_standings(matches: list[dict]) -> list[dict]:
    standings: dict[str, dict] = {}

    for match in matches:
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        home_score = match.get("home_score")
        away_score = match.get("away_score")

        if not home_team or not away_team:
            continue
        if home_score is None or away_score is None:
            continue

        if home_team not in standings:
            standings[home_team] = {
                "team": home_team,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "points": 0,
            }
        if away_team not in standings:
            standings[away_team] = {
                "team": away_team,
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_for": 0,
                "goals_against": 0,
                "goal_difference": 0,
                "points": 0,
            }

        home = standings[home_team]
        away = standings[away_team]

        home["played"] += 1
        away["played"] += 1
        home["goals_for"] += home_score
        home["goals_against"] += away_score
        away["goals_for"] += away_score
        away["goals_against"] += home_score

        if home_score > away_score:
            home["wins"] += 1
            home["points"] += 3
            away["losses"] += 1
        elif home_score < away_score:
            away["wins"] += 1
            away["points"] += 3
            home["losses"] += 1
        else:
            home["draws"] += 1
            away["draws"] += 1
            home["points"] += 1
            away["points"] += 1

    table = []
    for row in standings.values():
        row["goal_difference"] = row["goals_for"] - row["goals_against"]
        table.append(row)

    table.sort(
        key=lambda r: (r["points"], r["goal_difference"], r["goals_for"]),
        reverse=True,
    )

    ranked = []
    for idx, row in enumerate(table, start=1):
        ranked.append(
            {
                "rank": idx,
                "team": row["team"],
                "played": row["played"],
                "wins": row["wins"],
                "draws": row["draws"],
                "losses": row["losses"],
                "goals_for": row["goals_for"],
                "goals_against": row["goals_against"],
                "goal_difference": row["goal_difference"],
                "points": row["points"],
            }
        )
    return ranked


def run_monte_carlo_simulation(
    teams: list[str],
    upcoming_probabilities: list[dict],
    base_points: dict[str, int],
    base_goal_difference: dict[str, int],
    base_goals_for: dict[str, int],
    num_simulations: int,
    seed: int | None = None,
) -> dict[str, list[int]]:
    """Run Monte Carlo season outcomes using fixed per-match probabilities."""
    rng = random.Random(seed)
    position_counts = {team: [0] * len(teams) for team in teams}

    for _ in range(num_simulations):
        sim_points = base_points.copy()
        for match in upcoming_probabilities:
            home_team = match["home_team"]
            away_team = match["away_team"]
            p_home = match["p_home"]
            p_draw = match["p_draw"]
            u = rng.random()
            if u < p_home:
                sim_points[home_team] += 3
            elif u < p_home + p_draw:
                sim_points[home_team] += 1
                sim_points[away_team] += 1
            else:
                sim_points[away_team] += 3

        final_order = sorted(
            teams,
            key=lambda t: (
                -sim_points[t],
                -base_goal_difference[t],
                -base_goals_for[t],
                t,
            ),
        )
        for pos_idx, team in enumerate(final_order):
            position_counts[team][pos_idx] += 1

    return position_counts


def compute_season_context(
    stored_matches: list[dict],
    starting_ratings: dict[str, float],
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> dict:
    """Compute all season-scoped derived structures used by UI tabs."""
    finished_matches, completed_matches, upcoming_matches = split_matches_by_status(stored_matches)
    ratings, pregame_ratings = compute_elo_ratings(
        finished_matches,
        starting_ratings,
        include_pregame=True,
        home_advantage=home_advantage,
    )

    probabilities_table = []
    upcoming_probabilities = []
    for match in upcoming_matches:
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        utc_date = match.get("utc_date")
        matchday = match.get("matchday")
        if not home_team or not away_team:
            continue
        p_home, p_draw, p_away = predict_match(home_team, away_team, ratings, home_advantage=home_advantage)
        home_elo = ratings.get(home_team, DEFAULT_ELO)
        away_elo = ratings.get(away_team, DEFAULT_ELO)
        upcoming_probabilities.append(
            {
                "home_team": home_team,
                "away_team": away_team,
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
            }
        )
        probabilities_table.append(
            {
                "date": format_display_date(utc_date),
                "matchday": matchday,
                "home_team": home_team,
                "home_elo": round(home_elo),
                "away_team": away_team,
                "away_elo": round(away_elo),
                "p_home": f"{p_home * 100:.1f}",
                "p_draw": f"{p_draw * 100:.1f}",
                "p_away": f"{p_away * 100:.1f}",
            }
        )

    completed_table = []
    for match in completed_matches:
        match_id = match.get("match_id")
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        pregame = pregame_ratings.get(match_id, {})
        pregame_home_elo = pregame.get("pregame_home_elo", starting_ratings.get(home_team, DEFAULT_ELO))
        pregame_away_elo = pregame.get("pregame_away_elo", starting_ratings.get(away_team, DEFAULT_ELO))
        postgame_home_elo = pregame.get("postgame_home_elo", pregame_home_elo)
        completed_table.append(
            {
                "matchday": match.get("matchday"),
                "date": format_display_date(match.get("utc_date")),
                "home_team": home_team,
                "pregame_home_elo": round(pregame_home_elo),
                "away_team": away_team,
                "pregame_away_elo": round(pregame_away_elo),
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
                "home_elo_change": round(postgame_home_elo - pregame_home_elo),
            }
        )

    elo_change_values = {team: rating - starting_ratings.get(team, DEFAULT_ELO) for team, rating in ratings.items()}
    elo_change_rank = {
        team: rank
        for rank, (team, _) in enumerate(
            sorted(elo_change_values.items(), key=lambda item: item[1], reverse=True),
            start=1,
        )
    }
    ratings_table = [
        {
            "elo_rank": rank,
            "team": team,
            "rating": round(rating),
            "elo_change": round(elo_change_values[team]),
            "elo_change_rank": int(elo_change_rank[team]),
        }
        for rank, (team, rating) in enumerate(
            sorted(ratings.items(), key=lambda item: item[1], reverse=True),
            start=1,
        )
    ]

    standings_table = compute_league_standings(finished_matches)

    return {
        "finished_matches": finished_matches,
        "completed_matches": completed_matches,
        "upcoming_matches": upcoming_matches,
        "ratings": ratings,
        "pregame_ratings": pregame_ratings,
        "probabilities_table": probabilities_table,
        "upcoming_probabilities": upcoming_probabilities,
        "completed_table": completed_table,
        "ratings_table": ratings_table,
        "standings_table": standings_table,
    }

st.set_page_config(page_title="Football League Forecasting Tool", page_icon="⚽", layout="wide")

st.title("Football League Forecast Lab")

competition_label = st.selectbox(
    "Competition",
    options=list(COMPETITION_OPTIONS.keys()),
    index=0,
)
competition = COMPETITION_OPTIONS[competition_label]
season = st.selectbox("Season", options=SEASON_OPTIONS, index=0)
data_last_updated_display = "Unavailable"
try:
    fetched_matches, fetched_at_utc = fetch_matches_cached(competition=competition, season=int(season))
    save_matches(fetched_matches, competition=competition, season=int(season))
    fetched_at_dt = datetime.fromisoformat(fetched_at_utc)
    data_last_updated_display = fetched_at_dt.strftime("%b %d, %Y %H:%M UTC")
except Exception:
    st.warning("Failed to refresh API data. Showing most recent local data if available; it may be stale.")
    data_last_updated_display = "Unavailable (using local DB fallback)"

st.caption(f"Data last updated: {data_last_updated_display} (refreshes hourly)")

stored_matches = get_matches(competition=competition, season=int(season))

starting_ratings, starting_ratings_info = load_starting_ratings_csv(competition=competition)
if starting_ratings_info:
    st.info(starting_ratings_info)

ha_map, ha_warning = load_home_advantage_config()
if ha_warning:
    st.warning(ha_warning)
home_advantage = ha_map.get(competition, DEFAULT_HOME_ADVANTAGE)
ha_source = "from config" if competition in ha_map else "default"
st.caption(f"Home advantage: {home_advantage:g} ({ha_source})")

season_context = compute_season_context(stored_matches, starting_ratings, home_advantage=home_advantage)
finished_matches = season_context["finished_matches"]
probabilities_table = season_context["probabilities_table"]
completed_table = season_context["completed_table"]
ratings_table = season_context["ratings_table"]
standings_table = season_context["standings_table"]
upcoming_probabilities = season_context["upcoming_probabilities"]
ratings = season_context["ratings"]
completed_matches = season_context["completed_matches"]
upcoming_matches = season_context["upcoming_matches"]
pregame_ratings = season_context["pregame_ratings"]

data_elo_tab, simulations_tab, team_deep_dive_tab, diagnostics_tab = st.tabs(
    ["Data & Elo", "Simulations", "Team Deep Dive", "Diagnostics"]
)

with data_elo_tab:
    st.subheader("Upcoming Match Probabilities")
    if probabilities_table:
        st.dataframe(probabilities_table, use_container_width=True)
    else:
        st.info(NO_UPCOMING_MSG)

    st.subheader("Completed matches")
    if completed_table:
        st.dataframe(completed_table, use_container_width=True)
    else:
        st.info(NO_COMPLETED_MSG)

    st.subheader("Elo Ratings")
    if ratings_table:
        ratings_display_df = clean_display_df(pd.DataFrame(ratings_table))
        ratings_height = full_table_height(ratings_display_df)
        st.dataframe(
            ratings_display_df,
            hide_index=True,
            height=ratings_height,
            use_container_width=True,
        )
    else:
        st.info(NO_RATINGS_MSG)

    st.subheader("League standings")
    if standings_table:
        standings_display_df = clean_display_df(pd.DataFrame(standings_table))
        standings_height = full_table_height(standings_display_df)
        st.dataframe(
            standings_display_df,
            hide_index=True,
            height=standings_height,
            use_container_width=True,
        )
    else:
        st.info(NO_STANDINGS_MSG)

with simulations_tab:
    st.header("Monte Carlo Simulation")
    num_simulations = st.number_input(
        "Number of simulations",
        min_value=MONTE_CARLO_MIN,
        max_value=MONTE_CARLO_MAX,
        value=MONTE_CARLO_DEFAULT,
        step=MONTE_CARLO_STEP,
    )
    seed_text = st.text_input("Random seed (optional)", value="")
    run_simulation = st.button("Run simulation")

    if run_simulation:
        seed_value = None
        if seed_text.strip():
            try:
                seed_value = int(seed_text.strip())
            except ValueError:
                st.info("Invalid seed. Using non-fixed randomness.")
                seed_value = None

        current_standings = season_context["standings_table"]
        standings_by_team = {row["team"]: row for row in current_standings}
        teams_set = set(standings_by_team.keys())
        for match in upcoming_probabilities:
            teams_set.add(match["home_team"])
            teams_set.add(match["away_team"])

        teams = sorted(teams_set)
        base_points = {team: standings_by_team.get(team, {}).get("points", 0) for team in teams}
        base_goal_difference = {
            team: standings_by_team.get(team, {}).get("goal_difference", 0) for team in teams
        }
        base_goals_for = {team: standings_by_team.get(team, {}).get("goals_for", 0) for team in teams}

        position_counts = run_monte_carlo_simulation(
            teams=teams,
            upcoming_probabilities=upcoming_probabilities,
            base_points=base_points,
            base_goal_difference=base_goal_difference,
            base_goals_for=base_goals_for,
            num_simulations=int(num_simulations),
            seed=seed_value,
        )

        teams_sorted_for_output = sorted(
            teams,
            key=lambda team: (-starting_ratings.get(team, DEFAULT_ELO), team),
        )
        position_labels = [str(i) for i in range(1, len(teams) + 1)]

        simulation_rows = []
        for team in teams_sorted_for_output:
            row = {"team": team}
            for idx, label in enumerate(position_labels):
                row[label] = round((position_counts[team][idx] * 100.0) / int(num_simulations), 1)
            simulation_rows.append(row)

        simulation_df = pd.DataFrame(simulation_rows)
        st.session_state["simulation_position_matrix"] = simulation_df.copy()
        heatmap_df = simulation_df.melt(
            id_vars="team",
            var_name="position",
            value_name="percentage",
        )
        heatmap_df["position_num"] = heatmap_df["position"].astype(int)
        n_teams = len(teams_sorted_for_output)

        heatmap = (
            alt.Chart(heatmap_df)
            .mark_rect(stroke="#d9d9d9", strokeWidth=0.6, opacity=1.0)
            .encode(
                x=alt.X(
                    "position_num:O",
                    title="Position",
                    sort=list(range(1, n_teams + 1)),
                    axis=alt.Axis(values=list(range(1, n_teams + 1))),
                ),
                y=alt.Y(
                    "team:N",
                    sort=teams_sorted_for_output,
                    title="Team",
                    axis=alt.Axis(
                        labelLimit=2000,
                        labelOverlap=False,
                        labelAngle=0,
                        labelFontSize=11,
                    ),
                ),
                color=alt.condition(
                    alt.datum.percentage <= 0,
                    alt.value("#ffffff"),
                    alt.Color(
                        "percentage:Q",
                        title="Probability (%)",
                        scale=alt.Scale(domain=[0, 50], clamp=True, range=["#ffffff", "#08306b"]),
                        legend=None,
                    ),
                ),
                tooltip=[
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("position_num:O", title="Position"),
                    alt.Tooltip("percentage:Q", title="Probability (%)", format=".1f"),
                ],
            )
            .properties(
                width=900,
                height=max(500, 36 * n_teams),
                padding={"left": 160, "right": 10, "top": 10, "bottom": 40},
            )
        )
        st.altair_chart(heatmap, use_container_width=True)

        simulation_display_df = simulation_df.copy()
        probability_columns = [col for col in simulation_display_df.columns if col != "team"]
        simulation_display_df[probability_columns] = simulation_display_df[probability_columns].round(1)
        simulation_display_df = clean_display_df(simulation_display_df)
        simulation_height = full_table_height(simulation_display_df)
        simulation_column_config = {
            col: st.column_config.NumberColumn(format="%.1f")
            for col in probability_columns
            if col in simulation_display_df.columns
        }
        st.dataframe(
            simulation_display_df,
            hide_index=True,
            height=simulation_height,
            use_container_width=True,
            column_config=simulation_column_config,
        )

with team_deep_dive_tab:
    teams_for_select = sorted(
        list(ratings.keys()),
        key=lambda team: (-starting_ratings.get(team, DEFAULT_ELO), team),
    )
    if not teams_for_select:
        st.info("No teams available for deep dive.")
    else:
        selected_team = st.selectbox("Team", options=teams_for_select, index=0)

        current_elo = ratings.get(selected_team, DEFAULT_ELO)
        starting_elo = starting_ratings.get(selected_team, DEFAULT_ELO)
        elo_change = current_elo - starting_elo
        standings_lookup = {row["team"]: row for row in standings_table}
        team_standing = standings_lookup.get(selected_team, {})
        current_rank = team_standing.get("rank")
        current_points = team_standing.get("points", 0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Elo", f"{round(current_elo)}")
        c2.metric("Elo Change", f"{round(elo_change)}")
        c3.metric("League Rank / Points", f"{current_rank} / {current_points}" if current_rank else f"- / {current_points}")

        team_completed_matches = [
            m
            for m in completed_matches
            if m.get("home_team") == selected_team or m.get("away_team") == selected_team
        ]
        team_completed_matches = sorted(team_completed_matches, key=lambda m: m.get("utc_date") or "")

        elo_evolution_rows = []
        for match in team_completed_matches:
            match_id = match.get("match_id")
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            prepost = pregame_ratings.get(match_id, {})
            is_home = home_team == selected_team
            postgame_team_elo = (
                prepost.get("postgame_home_elo") if is_home else prepost.get("postgame_away_elo")
            )
            if postgame_team_elo is None:
                continue
            elo_evolution_rows.append(
                {
                    "utc_date": match.get("utc_date"),
                    "date": format_display_date(match.get("utc_date")),
                    "elo": postgame_team_elo,
                }
            )

        if elo_evolution_rows:
            elo_chart_df = pd.DataFrame(elo_evolution_rows)
            elo_chart_df["date_dt"] = pd.to_datetime(elo_chart_df["utc_date"], utc=True, errors="coerce")
            elo_chart_df = elo_chart_df.dropna(subset=["date_dt"]).sort_values("date_dt")
            if not elo_chart_df.empty:
                elo_min = float(elo_chart_df["elo"].min())
                elo_max = float(elo_chart_df["elo"].max())
                margin = max(5.0, (elo_max - elo_min) * 0.05)
                y_scale = alt.Scale(domain=[elo_min - margin, elo_max + margin])

                line = (
                    alt.Chart(elo_chart_df)
                    .mark_line(strokeWidth=1.5)
                    .encode(
                        x=alt.X("date_dt:T", title="Date", axis=alt.Axis(format="%b %d, %Y")),
                        y=alt.Y("elo:Q", title="Elo", scale=y_scale),
                        tooltip=[
                            alt.Tooltip("date:N", title="Date"),
                            alt.Tooltip("elo:Q", title="Elo", format=".1f"),
                        ],
                    )
                )
                points = (
                    alt.Chart(elo_chart_df)
                    .mark_circle(size=45)
                    .encode(
                        x=alt.X("date_dt:T"),
                        y=alt.Y("elo:Q", scale=y_scale),
                        tooltip=[
                            alt.Tooltip("date:N", title="Date"),
                            alt.Tooltip("elo:Q", title="Elo", format=".1f"),
                        ],
                    )
                )
                st.altair_chart((line + points).properties(height=320), use_container_width=True)
            else:
                st.info("No completed matches with valid dates for Elo evolution.")
        else:
            st.info("No completed matches available for Elo evolution.")

        team_completed_rows = []
        completed_points_total = 0
        completed_matches_count = 0
        for match in sorted(team_completed_matches, key=lambda m: m.get("utc_date") or "", reverse=True):
            match_id = match.get("match_id")
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            home_score = match.get("home_score")
            away_score = match.get("away_score")
            is_home = home_team == selected_team
            opponent = away_team if is_home else home_team
            home_away = "Home" if is_home else "Away"
            prepost = pregame_ratings.get(match_id, {})
            pregame_team_elo = prepost.get("pregame_home_elo") if is_home else prepost.get("pregame_away_elo")
            pregame_opponent_elo = prepost.get("pregame_away_elo") if is_home else prepost.get("pregame_home_elo")
            postgame_team_elo = prepost.get("postgame_home_elo") if is_home else prepost.get("postgame_away_elo")
            if pregame_team_elo is None:
                pregame_team_elo = starting_ratings.get(selected_team, DEFAULT_ELO)
            if pregame_opponent_elo is None:
                pregame_opponent_elo = starting_ratings.get(opponent, DEFAULT_ELO)
            if postgame_team_elo is None:
                postgame_team_elo = pregame_team_elo
            if home_score is not None and away_score is not None:
                completed_matches_count += 1
                if is_home:
                    if home_score > away_score:
                        completed_points_total += 3
                    elif home_score == away_score:
                        completed_points_total += 1
                else:
                    if away_score > home_score:
                        completed_points_total += 3
                    elif away_score == home_score:
                        completed_points_total += 1
            team_completed_rows.append(
                {
                    "matchday": match.get("matchday"),
                    "date": format_display_date(match.get("utc_date")),
                    "opponent": opponent,
                    "home_away": home_away,
                    "score": f"{home_score}-{away_score}",
                    "pregame_team_elo": round(pregame_team_elo),
                    "pregame_opponent_elo": round(pregame_opponent_elo),
                    "team_elo_change": round(postgame_team_elo - pregame_team_elo),
                }
            )

        completed_ppm_text = "N/A"
        if completed_matches_count > 0:
            completed_ppm_text = f"{(completed_points_total / completed_matches_count):.1f}"
        st.subheader(f"Completed Matches (PPM: {completed_ppm_text})")
        if team_completed_rows:
            st.dataframe(team_completed_rows, use_container_width=True)
        else:
            st.info("No completed matches for selected team.")

        team_upcoming_matches = [
            m
            for m in upcoming_matches
            if m.get("home_team") == selected_team or m.get("away_team") == selected_team
        ]
        team_upcoming_matches = sorted(
            team_upcoming_matches,
            key=lambda m: (m.get("matchday") if m.get("matchday") is not None else 9999, m.get("utc_date") or ""),
        )
        remaining_rows = []
        expected_points_total = 0.0
        remaining_fixtures_count = 0
        for match in team_upcoming_matches:
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            is_home = home_team == selected_team
            opponent = away_team if is_home else home_team
            home_away = "Home" if is_home else "Away"
            team_elo = ratings.get(selected_team, DEFAULT_ELO)
            opponent_elo = ratings.get(opponent, DEFAULT_ELO)
            p_home, p_draw, p_away = predict_match(home_team, away_team, ratings, home_advantage=home_advantage)
            p_win = p_home if is_home else p_away
            p_loss = p_away if is_home else p_home
            expected_points_total += (3.0 * p_win) + (1.0 * p_draw)
            remaining_fixtures_count += 1
            remaining_rows.append(
                {
                    "matchday": match.get("matchday"),
                    "date": format_display_date(match.get("utc_date")),
                    "opponent": opponent,
                    "home_away": home_away,
                    "team_elo": round(team_elo),
                    "opponent_elo": round(opponent_elo),
                    "p_win": round(p_win * 100, 1),
                    "p_draw": round(p_draw * 100, 1),
                    "p_loss": round(p_loss * 100, 1),
                }
            )

        expected_ppm_text = "N/A"
        if remaining_fixtures_count > 0:
            expected_ppm_text = f"{(expected_points_total / remaining_fixtures_count):.1f}"
        st.subheader(f"Remaining Fixtures (Expected PPM: {expected_ppm_text})")
        if remaining_rows:
            remaining_df = pd.DataFrame(remaining_rows)
            remaining_column_config = {
                col: st.column_config.NumberColumn(format="%.1f")
                for col in PROBABILITY_COLUMNS
            }
            st.dataframe(
                remaining_df,
                use_container_width=True,
                hide_index=True,
                column_config=remaining_column_config,
            )
        else:
            st.info("No remaining fixtures for selected team.")

with diagnostics_tab:
    aggregate_all_leagues = st.checkbox("Aggregate across all leagues", value=False)
    diagnostics_sources = []
    if aggregate_all_leagues:
        for comp_code in sorted(set(COMPETITION_OPTIONS.values())):
            comp_matches = get_matches(comp_code, int(season))
            comp_starting_ratings, _ = load_starting_ratings_csv(competition=comp_code)
            comp_home_advantage = ha_map.get(comp_code, DEFAULT_HOME_ADVANTAGE)
            comp_context = compute_season_context(
                comp_matches,
                comp_starting_ratings,
                home_advantage=comp_home_advantage,
            )
            diagnostics_sources.append(
                (
                    comp_context["finished_matches"],
                    comp_context["pregame_ratings"],
                    comp_starting_ratings,
                    comp_home_advantage,
                )
            )
    else:
        diagnostics_sources.append((finished_matches, pregame_ratings, starting_ratings, home_advantage))

    completed_pred_rows = []
    delta_analysis_rows = []
    skipped_missing_pregame = 0
    skipped_invalid_delta_probs = 0
    for source_finished_matches, source_pregame_ratings, source_starting_ratings, source_home_advantage in diagnostics_sources:
        for match in source_finished_matches:
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            home_score = match.get("home_score")
            away_score = match.get("away_score")
            match_id = match.get("match_id")
            if not home_team or not away_team:
                continue
            if home_score is None or away_score is None:
                continue

            prepost = source_pregame_ratings.get(match_id, {})
            pregame_home_elo = prepost.get("pregame_home_elo", source_starting_ratings.get(home_team, DEFAULT_ELO))
            pregame_away_elo = prepost.get("pregame_away_elo", source_starting_ratings.get(away_team, DEFAULT_ELO))
            p_home_win, p_draw, p_home_loss = predict_match(
                home_team,
                away_team,
                {
                    home_team: pregame_home_elo,
                    away_team: pregame_away_elo,
                },
                home_advantage=source_home_advantage,
            )

            completed_pred_rows.append(
                {
                    "matchday": match.get("matchday"),
                    "p_home_win": p_home_win,
                    "p_draw": p_draw,
                    "p_home_loss": p_home_loss,
                    "actual_home_win": 1 if home_score > away_score else 0,
                    "actual_draw": 1 if home_score == away_score else 0,
                    "actual_home_loss": 1 if home_score < away_score else 0,
                }
            )

            # Delta-binned diagnostics require pregame ratings explicitly from pregame map.
            if match_id not in source_pregame_ratings:
                skipped_missing_pregame += 1
                continue
            pregame_home_elo_delta = prepost.get("pregame_home_elo")
            pregame_away_elo_delta = prepost.get("pregame_away_elo")
            if pregame_home_elo_delta is None or pregame_away_elo_delta is None:
                skipped_missing_pregame += 1
                continue
            p_home_delta, p_draw_delta, p_away_delta = predict_match(
                home_team,
                away_team,
                {
                    home_team: pregame_home_elo_delta,
                    away_team: pregame_away_elo_delta,
                },
                home_advantage=source_home_advantage,
            )
            probs_delta = [p_home_delta, p_draw_delta, p_away_delta]
            if any(v is None for v in probs_delta):
                skipped_invalid_delta_probs += 1
                continue
            if any((not isinstance(v, (int, float))) for v in probs_delta):
                skipped_invalid_delta_probs += 1
                continue
            if any(math.isnan(float(v)) for v in probs_delta):
                skipped_invalid_delta_probs += 1
                continue
            delta_value = (pregame_home_elo_delta + source_home_advantage) - pregame_away_elo_delta
            if not isinstance(delta_value, (int, float)) or math.isnan(float(delta_value)):
                skipped_invalid_delta_probs += 1
                continue
            delta_analysis_rows.append(
                {
                    "delta": float(delta_value),
                    "p_home": float(p_home_delta),
                    "p_draw": float(p_draw_delta),
                    "p_away": float(p_away_delta),
                    "y_home": 1 if home_score > away_score else 0,
                    "y_draw": 1 if home_score == away_score else 0,
                    "y_away": 1 if home_score < away_score else 0,
                }
            )

    epsilon = 1e-15

    def clamp_and_renormalize_probs(p_home: float, p_draw: float, p_away: float) -> tuple[float, float, float]:
        vals = [
            max(epsilon, min(1.0 - epsilon, p_home)),
            max(epsilon, min(1.0 - epsilon, p_draw)),
            max(epsilon, min(1.0 - epsilon, p_away)),
        ]
        denom = sum(vals)
        return vals[0] / denom, vals[1] / denom, vals[2] / denom

    filtered_rows = []
    for row in completed_pred_rows:
        p_home = row.get("p_home_win")
        p_draw = row.get("p_draw")
        p_away = row.get("p_home_loss")
        y_home = row.get("actual_home_win")
        y_draw = row.get("actual_draw")
        y_away = row.get("actual_home_loss")
        probs = [p_home, p_draw, p_away]
        if any(v is None for v in probs):
            continue
        if any((not isinstance(v, (int, float))) for v in probs):
            continue
        if any((v < 0.0 or v > 1.0) for v in probs):
            continue
        expected_log_loss = None
        expected_brier = None
        prob_sum = float(p_home + p_draw + p_away)
        if math.isfinite(prob_sum) and prob_sum > 0.0:
            # Expected metrics are computed from a normalized probability vector to avoid
            # numerical issues when model probabilities do not sum exactly to 1.0.
            p_home_norm = float(p_home) / prob_sum
            p_draw_norm = float(p_draw) / prob_sum
            p_away_norm = float(p_away) / prob_sum
            probs_norm = [p_home_norm, p_draw_norm, p_away_norm]
            if all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in probs_norm):
                expected_log_loss = -sum(v * math.log(max(epsilon, v)) for v in probs_norm)
                expected_brier = 1.0 - sum(v * v for v in probs_norm)
        filtered_rows.append(
            {
                "matchday": row.get("matchday"),
                "p_home": float(p_home),
                "p_draw": float(p_draw),
                "p_away": float(p_away),
                "y_home": int(y_home),
                "y_draw": int(y_draw),
                "y_away": int(y_away),
                "expected_log_loss": expected_log_loss,
                "expected_brier": expected_brier,
            }
        )

    matches_evaluated = len(filtered_rows)
    log_loss_value = None
    brier_value = None
    uniform_log_loss = None
    uniform_brier = None
    prevalence_log_loss = None
    prevalence_brier = None
    observed_home_rate = None
    observed_draw_rate = None
    observed_away_rate = None
    expected_log_loss_value = None
    expected_brier_value = None

    if matches_evaluated > 0:
        # Model metrics
        model_ll_sum = 0.0
        model_bs_sum = 0.0
        for row in filtered_rows:
            p_home = row["p_home"]
            p_draw = row["p_draw"]
            p_away = row["p_away"]
            y_home = row["y_home"]
            y_draw = row["y_draw"]
            y_away = row["y_away"]
            p_true = p_home if y_home == 1 else (p_draw if y_draw == 1 else p_away)
            p_true = max(epsilon, min(1.0 - epsilon, p_true))
            model_ll_sum += -math.log(p_true)
            model_bs_sum += ((p_home - y_home) ** 2) + ((p_draw - y_draw) ** 2) + ((p_away - y_away) ** 2)
        log_loss_value = model_ll_sum / matches_evaluated
        brier_value = model_bs_sum / matches_evaluated

        # Uniform baseline metrics
        u_home, u_draw, u_away = clamp_and_renormalize_probs(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        uniform_ll_sum = 0.0
        uniform_bs_sum = 0.0
        for row in filtered_rows:
            y_home = row["y_home"]
            y_draw = row["y_draw"]
            y_away = row["y_away"]
            u_true = u_home if y_home == 1 else (u_draw if y_draw == 1 else u_away)
            uniform_ll_sum += -math.log(u_true)
            uniform_bs_sum += ((u_home - y_home) ** 2) + ((u_draw - y_draw) ** 2) + ((u_away - y_away) ** 2)
        uniform_log_loss = uniform_ll_sum / matches_evaluated
        uniform_brier = uniform_bs_sum / matches_evaluated

        # Prevalence baseline metrics
        k_home = sum(row["y_home"] for row in filtered_rows)
        k_draw = sum(row["y_draw"] for row in filtered_rows)
        k_away = sum(row["y_away"] for row in filtered_rows)
        q_home = k_home / matches_evaluated
        q_draw = k_draw / matches_evaluated
        q_away = k_away / matches_evaluated
        q_home, q_draw, q_away = clamp_and_renormalize_probs(q_home, q_draw, q_away)
        observed_home_rate = q_home
        observed_draw_rate = q_draw
        observed_away_rate = q_away

        prevalence_ll_sum = 0.0
        prevalence_bs_sum = 0.0
        for row in filtered_rows:
            y_home = row["y_home"]
            y_draw = row["y_draw"]
            y_away = row["y_away"]
            q_true = q_home if y_home == 1 else (q_draw if y_draw == 1 else q_away)
            prevalence_ll_sum += -math.log(q_true)
            prevalence_bs_sum += ((q_home - y_home) ** 2) + ((q_draw - y_draw) ** 2) + ((q_away - y_away) ** 2)
        prevalence_log_loss = prevalence_ll_sum / matches_evaluated
        prevalence_brier = prevalence_bs_sum / matches_evaluated

        expected_ll_values = [
            row["expected_log_loss"]
            for row in filtered_rows
            if isinstance(row.get("expected_log_loss"), (int, float))
            and math.isfinite(float(row["expected_log_loss"]))
        ]
        expected_bs_values = [
            row["expected_brier"]
            for row in filtered_rows
            if isinstance(row.get("expected_brier"), (int, float))
            and math.isfinite(float(row["expected_brier"]))
        ]
        if expected_ll_values:
            expected_log_loss_value = sum(expected_ll_values) / len(expected_ll_values)
        if expected_bs_values:
            expected_brier_value = sum(expected_bs_values) / len(expected_bs_values)

    log_loss_text = "N/A" if log_loss_value is None else f"{log_loss_value:.3f}"
    brier_text = "N/A" if brier_value is None else f"{brier_value:.3f}"
    uniform_log_loss_text = "N/A" if uniform_log_loss is None else f"{uniform_log_loss:.3f}"
    uniform_brier_text = "N/A" if uniform_brier is None else f"{uniform_brier:.3f}"
    prevalence_log_loss_text = "N/A" if prevalence_log_loss is None else f"{prevalence_log_loss:.3f}"
    prevalence_brier_text = "N/A" if prevalence_brier is None else f"{prevalence_brier:.3f}"
    expected_log_loss_text = "n/a" if expected_log_loss_value is None else f"{expected_log_loss_value:.3f}"
    expected_brier_text = "n/a" if expected_brier_value is None else f"{expected_brier_value:.3f}"
    st.write(f"Matches evaluated: {matches_evaluated}")
    st.write(
        f"Log Loss: {log_loss_text} (Uniform: {uniform_log_loss_text} | Prevalence: {prevalence_log_loss_text} | Expected: {expected_log_loss_text})"
    )
    st.write(
        f"Brier Score: {brier_text} (Uniform: {uniform_brier_text} | Prevalence: {prevalence_brier_text} | Expected: {expected_brier_text})"
    )
    if observed_home_rate is not None and observed_draw_rate is not None and observed_away_rate is not None:
        st.caption(
            f"Observed rates: Home W {observed_home_rate:.3f}, Draw {observed_draw_rate:.3f}, Home L {observed_away_rate:.3f}"
        )

    if not completed_pred_rows:
        st.info("No completed matches available for diagnostics.")
    else:
        completed_pred_df = pd.DataFrame(completed_pred_rows)
        calibration_specs = [
            (
                "Home Win Calibration",
                "p_home_win",
                "actual_home_win",
                CALIBRATION_FULL_BIN_START,
                CALIBRATION_FULL_BIN_END,
                CALIBRATION_FULL_BIN_WIDTH,
            ),
            (
                "Draw Calibration",
                "p_draw",
                "actual_draw",
                CALIBRATION_BIN_START,
                CALIBRATION_BIN_END,
                CALIBRATION_BIN_WIDTH,
            ),
            (
                "Home Loss Calibration",
                "p_home_loss",
                "actual_home_loss",
                CALIBRATION_FULL_BIN_START,
                CALIBRATION_FULL_BIN_END,
                CALIBRATION_FULL_BIN_WIDTH,
            ),
        ]
        c1, c2, c3 = st.columns(3)
        for col, (title, pred_col, actual_col, bin_start, bin_end, bin_width) in zip((c1, c2, c3), calibration_specs):
            bin_count = int(round((bin_end - bin_start) / bin_width))
            bin_edges = [
                round(bin_start + (i * bin_width), 10)
                for i in range(bin_count + 1)
            ]
            calibration_df = build_calibration_df(completed_pred_df, pred_col, actual_col, bin_edges)
            with col:
                st.subheader(title)
                if calibration_df.empty:
                    st.info("No calibration data in configured bins.")
                    continue

                reference_df = pd.DataFrame(
                    {
                        "x": [bin_start, bin_end],
                        "y": [bin_start, bin_end],
                    }
                )
                ref_line = (
                    alt.Chart(reference_df)
                    .mark_line(color="#888888", strokeDash=[4, 4])
                    .encode(
                        x=alt.X("x:Q", scale=alt.Scale(domain=[bin_start, bin_end])),
                        y=alt.Y("y:Q", title="Actual frequency", scale=alt.Scale(domain=[0.0, 1.0])),
                    )
                )
                cal_error = (
                    alt.Chart(calibration_df)
                    .mark_errorbar(color="#6b6b6b")
                    .encode(
                        x=alt.X("bin_center:Q"),
                        y=alt.Y(
                            "ci_low:Q",
                            title="Actual frequency",
                            scale=alt.Scale(domain=[0.0, 1.0]),
                        ),
                        y2=alt.Y2("ci_high:Q"),
                        tooltip=[
                            alt.Tooltip("bin_label:N", title="Bin"),
                            alt.Tooltip("bin_center:Q", title="Predicted", format=".2f"),
                            alt.Tooltip("actual_rate:Q", title="Actual", format=".2f"),
                            alt.Tooltip("ci_low:Q", title="95% CI Low", format=".2f"),
                            alt.Tooltip("ci_high:Q", title="95% CI High", format=".2f"),
                            alt.Tooltip("n:Q", title="n"),
                        ],
                    )
                )
                cal_line = (
                    alt.Chart(calibration_df)
                    .mark_line()
                    .encode(
                        x=alt.X(
                            "bin_center:Q",
                            title="Predicted probability",
                            scale=alt.Scale(domain=[bin_start, bin_end]),
                        ),
                        y=alt.Y(
                            "actual_rate:Q",
                            title="Actual frequency",
                            scale=alt.Scale(domain=[0.0, 1.0]),
                        ),
                        tooltip=[
                            alt.Tooltip("bin_label:N", title="Bin"),
                            alt.Tooltip("bin_center:Q", title="Predicted", format=".2f"),
                            alt.Tooltip("actual_rate:Q", title="Actual", format=".2f"),
                            alt.Tooltip("ci_low:Q", title="95% CI Low", format=".2f"),
                            alt.Tooltip("ci_high:Q", title="95% CI High", format=".2f"),
                            alt.Tooltip("n:Q", title="n"),
                        ],
                    )
                )
                cal_points = (
                    alt.Chart(calibration_df)
                    .mark_circle(size=55)
                    .encode(
                        x=alt.X("bin_center:Q", scale=alt.Scale(domain=[bin_start, bin_end])),
                        y=alt.Y(
                            "actual_rate:Q",
                            title="Actual frequency",
                            scale=alt.Scale(domain=[0.0, 1.0]),
                        ),
                        tooltip=[
                            alt.Tooltip("bin_label:N", title="Bin"),
                            alt.Tooltip("bin_center:Q", title="Predicted", format=".2f"),
                            alt.Tooltip("actual_rate:Q", title="Actual", format=".2f"),
                            alt.Tooltip("ci_low:Q", title="95% CI Low", format=".2f"),
                            alt.Tooltip("ci_high:Q", title="95% CI High", format=".2f"),
                            alt.Tooltip("n:Q", title="n"),
                        ],
                    )
                )
                chart = (ref_line + cal_error + cal_line + cal_points).resolve_scale(x="shared", y="shared")
                st.altair_chart(chart.properties(height=280), use_container_width=True)

        st.subheader("Calibration by Elo Delta (pregame Elo + home advantage)")
        if skipped_missing_pregame > 0 or skipped_invalid_delta_probs > 0:
            st.caption(
                f"Delta calibration skips: missing pregame={skipped_missing_pregame}, invalid probability/delta={skipped_invalid_delta_probs}"
            )
        if len(delta_analysis_rows) < 10:
            st.warning("Too few completed matches for delta-binned calibration (<10).")
        else:
            delta_df = pd.DataFrame(delta_analysis_rows)

            def delta_bin_info(delta: float) -> tuple[str, int]:
                if delta < -200:
                    return "< -200", 0
                if delta < -100:
                    return "-200 to -100", 1
                if delta < -50:
                    return "-100 to -50", 2
                if delta < 50:
                    return "-50 to 50", 3
                if delta < 100:
                    return "50 to 100", 4
                if delta < 200:
                    return "100 to 200", 5
                return "> 200", 6

            delta_df["bin_label"] = delta_df["delta"].apply(lambda d: delta_bin_info(d)[0])
            delta_df["bin_order"] = delta_df["delta"].apply(lambda d: delta_bin_info(d)[1])
            bin_order_labels = ["< -200", "-200 to -100", "-100 to -50", "-50 to 50", "50 to 100", "100 to 200", "> 200"]

            delta_agg = (
                delta_df.groupby(["bin_label", "bin_order"], as_index=False)
                .agg(
                    n=("delta", "size"),
                    pred_home=("p_home", "mean"),
                    pred_draw=("p_draw", "mean"),
                    pred_away=("p_away", "mean"),
                    k_home=("y_home", "sum"),
                    k_draw=("y_draw", "sum"),
                    k_away=("y_away", "sum"),
                )
                .sort_values("bin_order")
            )
            delta_agg["obs_home"] = delta_agg["k_home"] / delta_agg["n"]
            delta_agg["obs_draw"] = delta_agg["k_draw"] / delta_agg["n"]
            delta_agg["obs_away"] = delta_agg["k_away"] / delta_agg["n"]
            delta_agg[["obs_home_lo", "obs_home_hi"]] = delta_agg.apply(
                lambda r: pd.Series(wilson_ci(int(r["k_home"]), int(r["n"]), z=1.96)),
                axis=1,
            )
            delta_agg[["obs_draw_lo", "obs_draw_hi"]] = delta_agg.apply(
                lambda r: pd.Series(wilson_ci(int(r["k_draw"]), int(r["n"]), z=1.96)),
                axis=1,
            )
            delta_agg[["obs_away_lo", "obs_away_hi"]] = delta_agg.apply(
                lambda r: pd.Series(wilson_ci(int(r["k_away"]), int(r["n"]), z=1.96)),
                axis=1,
            )

            chart_specs = [
                ("Home Win", "pred_home", "obs_home", "obs_home_lo", "obs_home_hi"),
                ("Draw", "pred_draw", "obs_draw", "obs_draw_lo", "obs_draw_hi"),
                ("Home Loss", "pred_away", "obs_away", "obs_away_lo", "obs_away_hi"),
            ]
            d1, d2, d3 = st.columns(3)
            for col, (title, pred_col, obs_col, lo_col, hi_col) in zip((d1, d2, d3), chart_specs):
                with col:
                    st.subheader(title)
                    pred_obs_long = pd.concat(
                        [
                            delta_agg[["bin_label", "bin_order", "n", pred_col]].rename(
                                columns={pred_col: "value"}
                            ).assign(series_name="Predicted"),
                            delta_agg[["bin_label", "bin_order", "n", obs_col]].rename(
                                columns={obs_col: "value"}
                            ).assign(series_name="Observed"),
                        ],
                        ignore_index=True,
                    )

                    line = (
                        alt.Chart(pred_obs_long)
                        .mark_line()
                        .encode(
                            x=alt.X("bin_label:N", title="Elo delta bin", sort=bin_order_labels),
                            y=alt.Y("value:Q", title="Probability", scale=alt.Scale(domain=[0.0, 1.0])),
                            color=alt.Color("series_name:N", title="Series"),
                            detail="series_name:N",
                            tooltip=[
                                alt.Tooltip("bin_label:N", title="Bin"),
                                alt.Tooltip("series_name:N", title="Series"),
                                alt.Tooltip("value:Q", title="Value", format=".3f"),
                                alt.Tooltip("n:Q", title="n"),
                            ],
                        )
                    )
                    points = (
                        alt.Chart(pred_obs_long)
                        .mark_circle(size=55)
                        .encode(
                            x=alt.X("bin_label:N", sort=bin_order_labels),
                            y=alt.Y("value:Q", scale=alt.Scale(domain=[0.0, 1.0])),
                            color=alt.Color("series_name:N", title="Series"),
                            tooltip=[
                                alt.Tooltip("bin_label:N", title="Bin"),
                                alt.Tooltip("series_name:N", title="Series"),
                                alt.Tooltip("value:Q", title="Value", format=".3f"),
                                alt.Tooltip("n:Q", title="n"),
                            ],
                        )
                    )
                    obs_error = (
                        alt.Chart(delta_agg)
                        .mark_errorbar(color="#6b6b6b")
                        .encode(
                            x=alt.X("bin_label:N", sort=bin_order_labels),
                            y=alt.Y(lo_col + ":Q", scale=alt.Scale(domain=[0.0, 1.0])),
                            y2=alt.Y2(hi_col + ":Q"),
                            tooltip=[
                                alt.Tooltip("bin_label:N", title="Bin"),
                                alt.Tooltip("n:Q", title="n"),
                                alt.Tooltip(obs_col + ":Q", title="Observed", format=".3f"),
                                alt.Tooltip(lo_col + ":Q", title="95% CI Low", format=".3f"),
                                alt.Tooltip(hi_col + ":Q", title="95% CI High", format=".3f"),
                            ],
                        )
                    )
                    st.altair_chart((obs_error + line + points).properties(height=280), use_container_width=True)

        # Matchday performance charts (exclude rows without matchday only for these charts).
        matchday_metric_buckets: dict[int, dict[str, float]] = {}
        for row in filtered_rows:
            matchday = row.get("matchday")
            if matchday is None:
                continue
            if not isinstance(matchday, (int, float)):
                continue
            matchday_int = int(matchday)
            p_home = row["p_home"]
            p_draw = row["p_draw"]
            p_away = row["p_away"]
            y_home = row["y_home"]
            y_draw = row["y_draw"]
            y_away = row["y_away"]
            p_true = p_home if y_home == 1 else (p_draw if y_draw == 1 else p_away)
            p_true = max(epsilon, min(1.0 - epsilon, p_true))
            ll_i = -math.log(p_true)
            bs_i = ((p_home - y_home) ** 2) + ((p_draw - y_draw) ** 2) + ((p_away - y_away) ** 2)
            q_true = q_home if y_home == 1 else (q_draw if y_draw == 1 else q_away)
            ll_prev_i = -math.log(q_true)
            bs_prev_i = ((q_home - y_home) ** 2) + ((q_draw - y_draw) ** 2) + ((q_away - y_away) ** 2)

            if matchday_int not in matchday_metric_buckets:
                matchday_metric_buckets[matchday_int] = {
                    "model_ll_sum": 0.0,
                    "model_bs_sum": 0.0,
                    "prev_ll_sum": 0.0,
                    "prev_bs_sum": 0.0,
                    "expected_ll_sum": 0.0,
                    "expected_bs_sum": 0.0,
                    "expected_n": 0,
                    "n": 0,
                }
            matchday_metric_buckets[matchday_int]["model_ll_sum"] += ll_i
            matchday_metric_buckets[matchday_int]["model_bs_sum"] += bs_i
            matchday_metric_buckets[matchday_int]["prev_ll_sum"] += ll_prev_i
            matchday_metric_buckets[matchday_int]["prev_bs_sum"] += bs_prev_i
            expected_ll_i = row.get("expected_log_loss")
            expected_bs_i = row.get("expected_brier")
            if (
                isinstance(expected_ll_i, (int, float))
                and math.isfinite(float(expected_ll_i))
                and isinstance(expected_bs_i, (int, float))
                and math.isfinite(float(expected_bs_i))
            ):
                matchday_metric_buckets[matchday_int]["expected_ll_sum"] += float(expected_ll_i)
                matchday_metric_buckets[matchday_int]["expected_bs_sum"] += float(expected_bs_i)
                matchday_metric_buckets[matchday_int]["expected_n"] += 1
            matchday_metric_buckets[matchday_int]["n"] += 1

        if (
            matchday_metric_buckets
            and uniform_log_loss is not None
            and prevalence_log_loss is not None
            and q_home is not None
            and q_draw is not None
            and q_away is not None
        ):
            matchday_rows = []
            for md in sorted(matchday_metric_buckets.keys()):
                n_md = int(matchday_metric_buckets[md]["n"])
                ll_avg = matchday_metric_buckets[md]["model_ll_sum"] / n_md
                bs_avg = matchday_metric_buckets[md]["model_bs_sum"] / n_md
                prev_ll_avg = matchday_metric_buckets[md]["prev_ll_sum"] / n_md
                prev_bs_avg = matchday_metric_buckets[md]["prev_bs_sum"] / n_md
                expected_n_md = int(matchday_metric_buckets[md]["expected_n"])
                expected_ll_avg = (
                    matchday_metric_buckets[md]["expected_ll_sum"] / expected_n_md
                    if expected_n_md > 0
                    else float("nan")
                )
                expected_bs_avg = (
                    matchday_metric_buckets[md]["expected_bs_sum"] / expected_n_md
                    if expected_n_md > 0
                    else float("nan")
                )
                matchday_rows.append(
                    {
                        "matchday": md,
                        "n": n_md,
                        "model_log_loss": ll_avg,
                        "model_brier": bs_avg,
                        "prevalence_log_loss": prev_ll_avg,
                        "prevalence_brier": prev_bs_avg,
                        "uniform_log_loss": uniform_log_loss,
                        "uniform_brier": uniform_brier,
                        "expected_log_loss": expected_ll_avg,
                        "expected_brier": expected_bs_avg,
                        "expected_n": expected_n_md,
                    }
                )

            matchday_df = pd.DataFrame(matchday_rows)
            chart_matchday_rows = [
                row for row in matchday_rows
                if row.get("n", 0) >= 5
            ]
            log_loss_long_rows = []
            brier_long_rows = []
            for row in chart_matchday_rows:
                md = row["matchday"]
                n_md = row["n"]
                log_loss_long_rows.extend(
                    [
                        {"matchday": md, "metric_value": row["model_log_loss"], "series_name": "Model", "n": n_md},
                        {"matchday": md, "metric_value": row["prevalence_log_loss"], "series_name": "Prevalence", "n": n_md},
                        {"matchday": md, "metric_value": row["uniform_log_loss"], "series_name": "Uniform", "n": n_md},
                        {
                            "matchday": md,
                            "metric_value": row["expected_log_loss"],
                            "series_name": "Expected",
                            "n": row.get("expected_n"),
                        },
                    ]
                )
                brier_long_rows.extend(
                    [
                        {"matchday": md, "metric_value": row["model_brier"], "series_name": "Model", "n": n_md},
                        {"matchday": md, "metric_value": row["prevalence_brier"], "series_name": "Prevalence", "n": n_md},
                        {"matchday": md, "metric_value": row["uniform_brier"], "series_name": "Uniform", "n": n_md},
                        {
                            "matchday": md,
                            "metric_value": row["expected_brier"],
                            "series_name": "Expected",
                            "n": row.get("expected_n"),
                        },
                    ]
                )
            log_loss_long_df = pd.DataFrame(log_loss_long_rows)
            brier_long_df = pd.DataFrame(brier_long_rows)

            st.subheader("Matchday Performance")
            eligible_matchdays_ll = [
                row for row in matchday_rows
                if row.get("n", 0) >= 5
            ]
            denom_ll = len(eligible_matchdays_ll)
            if denom_ll > 0:
                num_ll_uniform = sum(
                    1 for row in eligible_matchdays_ll
                    if row["model_log_loss"] < row["uniform_log_loss"]
                )
                num_ll_prev = sum(
                    1 for row in eligible_matchdays_ll
                    if row["model_log_loss"] < row["prevalence_log_loss"]
                )
                num_ll_expected = sum(
                    1 for row in eligible_matchdays_ll
                    if isinstance(row.get("expected_log_loss"), (int, float))
                    and math.isfinite(float(row["expected_log_loss"]))
                    and row["model_log_loss"] < row["expected_log_loss"]
                )
                pct_ll_uniform = round((100.0 * num_ll_uniform) / denom_ll)
                pct_ll_prev = round((100.0 * num_ll_prev) / denom_ll)
                pct_ll_expected = round((100.0 * num_ll_expected) / denom_ll)
                ll_uniform_text = f"Model beats Uniform: {num_ll_uniform}/{denom_ll} ({pct_ll_uniform}%)"
                ll_prev_text = f"Model beats Prevalence: {num_ll_prev}/{denom_ll} ({pct_ll_prev}%)"
                ll_expected_text = f"Model beats Expected: {num_ll_expected}/{denom_ll} ({pct_ll_expected}%)"
            else:
                ll_uniform_text = "Model beats Uniform: N/A"
                ll_prev_text = "Model beats Prevalence: N/A"
                ll_expected_text = "Model beats Expected: N/A"

            eligible_matchdays_bs = [
                row for row in matchday_rows
                if row.get("n", 0) >= 5
            ]
            denom_bs = len(eligible_matchdays_bs)
            if denom_bs > 0:
                num_bs_uniform = sum(
                    1 for row in eligible_matchdays_bs
                    if row["model_brier"] < row["uniform_brier"]
                )
                num_bs_prev = sum(
                    1 for row in eligible_matchdays_bs
                    if row["model_brier"] < row["prevalence_brier"]
                )
                num_bs_expected = sum(
                    1 for row in eligible_matchdays_bs
                    if isinstance(row.get("expected_brier"), (int, float))
                    and math.isfinite(float(row["expected_brier"]))
                    and row["model_brier"] < row["expected_brier"]
                )
                pct_bs_uniform = round((100.0 * num_bs_uniform) / denom_bs)
                pct_bs_prev = round((100.0 * num_bs_prev) / denom_bs)
                pct_bs_expected = round((100.0 * num_bs_expected) / denom_bs)
                bs_uniform_text = f"Model beats Uniform: {num_bs_uniform}/{denom_bs} ({pct_bs_uniform}%)"
                bs_prev_text = f"Model beats Prevalence: {num_bs_prev}/{denom_bs} ({pct_bs_prev}%)"
                bs_expected_text = f"Model beats Expected: {num_bs_expected}/{denom_bs} ({pct_bs_expected}%)"
            else:
                bs_uniform_text = "Model beats Uniform: N/A"
                bs_prev_text = "Model beats Prevalence: N/A"
                bs_expected_text = "Model beats Expected: N/A"

            md_c1, md_c2 = st.columns(2)
            series_domain = ["Uniform", "Prevalence", "Model", "Expected"]
            series_colors = ["#808080", "#8ecae6", "#d62728", "#1d3557"]
            with md_c1:
                st.write(ll_uniform_text)
                st.write(ll_prev_text)
                st.write(ll_expected_text)
                st.subheader("Matchday Log Loss")
                if log_loss_long_df.empty:
                    st.info("No matchdays with at least 5 completed matches.")
                else:
                    ll_line = (
                    alt.Chart(log_loss_long_df)
                    .mark_line()
                    .encode(
                        x=alt.X("matchday:Q", title="Matchday"),
                        y=alt.Y("metric_value:Q", title="Log Loss"),
                        color=alt.Color(
                            "series_name:N",
                            title="Series",
                            scale=alt.Scale(domain=series_domain, range=series_colors),
                            sort=series_domain,
                        ),
                        tooltip=[
                            alt.Tooltip("matchday:Q", title="Matchday"),
                            alt.Tooltip("series_name:N", title="Series"),
                            alt.Tooltip("metric_value:Q", title="Value", format=".3f"),
                            alt.Tooltip("n:Q", title="n"),
                        ],
                    )
                )
                    ll_points = (
                    alt.Chart(log_loss_long_df)
                    .mark_circle(size=50)
                    .encode(
                        x=alt.X("matchday:Q"),
                        y=alt.Y("metric_value:Q"),
                        color=alt.Color(
                            "series_name:N",
                            title="Series",
                            scale=alt.Scale(domain=series_domain, range=series_colors),
                            sort=series_domain,
                        ),
                        tooltip=[
                            alt.Tooltip("matchday:Q", title="Matchday"),
                            alt.Tooltip("series_name:N", title="Series"),
                            alt.Tooltip("metric_value:Q", title="Value", format=".3f"),
                            alt.Tooltip("n:Q", title="n"),
                        ],
                    )
                )
                    st.altair_chart((ll_line + ll_points).properties(height=280), use_container_width=True)

            with md_c2:
                st.write(bs_uniform_text)
                st.write(bs_prev_text)
                st.write(bs_expected_text)
                st.subheader("Matchday Brier Score")
                if brier_long_df.empty:
                    st.info("No matchdays with at least 5 completed matches.")
                else:
                    bs_line = (
                    alt.Chart(brier_long_df)
                    .mark_line()
                    .encode(
                        x=alt.X("matchday:Q", title="Matchday"),
                        y=alt.Y("metric_value:Q", title="Brier Score"),
                        color=alt.Color(
                            "series_name:N",
                            title="Series",
                            scale=alt.Scale(domain=series_domain, range=series_colors),
                            sort=series_domain,
                        ),
                        tooltip=[
                            alt.Tooltip("matchday:Q", title="Matchday"),
                            alt.Tooltip("series_name:N", title="Series"),
                            alt.Tooltip("metric_value:Q", title="Value", format=".3f"),
                            alt.Tooltip("n:Q", title="n"),
                        ],
                    )
                )
                    bs_points = (
                    alt.Chart(brier_long_df)
                    .mark_circle(size=50)
                    .encode(
                        x=alt.X("matchday:Q"),
                        y=alt.Y("metric_value:Q"),
                        color=alt.Color(
                            "series_name:N",
                            title="Series",
                            scale=alt.Scale(domain=series_domain, range=series_colors),
                            sort=series_domain,
                        ),
                        tooltip=[
                            alt.Tooltip("matchday:Q", title="Matchday"),
                            alt.Tooltip("series_name:N", title="Series"),
                            alt.Tooltip("metric_value:Q", title="Value", format=".3f"),
                            alt.Tooltip("n:Q", title="n"),
                        ],
                    )
                )
                    st.altair_chart((bs_line + bs_points).properties(height=280), use_container_width=True)

st.divider()
for line in FOOTER_LINES:
    st.caption(line)
