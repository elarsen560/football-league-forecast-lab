import csv
import random
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from data_client import fetch_matches
from db import get_matches, init_db, save_matches
from elo import compute_elo_ratings, predict_match

init_db()


def format_display_date(utc_date: str) -> str:
    if not utc_date:
        return ""
    try:
        return datetime.fromisoformat(utc_date.replace("Z", "+00:00")).strftime("%b %d, %Y")
    except ValueError:
        return utc_date


def load_starting_ratings_csv(path: str = "starting_elo.csv") -> tuple[dict[str, float], str | None]:
    starting_ratings: dict[str, float] = {}
    try:
        with open(path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                team = row.get("team")
                rating = row.get("rating")
                if team is None or rating is None:
                    raise ValueError("CSV must include team and rating columns")
                starting_ratings[team] = float(rating)
        return starting_ratings, None
    except FileNotFoundError:
        return {}, "starting_elo.csv not found. Using default starting Elo ratings (1500)."
    except Exception:
        return {}, "Failed to parse starting_elo.csv. Using default starting Elo ratings (1500)."


def split_matches_by_status(stored_matches: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    finished_matches = [m for m in stored_matches if m.get("status") == "FINISHED"]
    finished_matches = sorted(finished_matches, key=lambda m: m.get("utc_date") or "")

    completed_matches = [m for m in stored_matches if m.get("status") == "FINISHED"]
    completed_matches = sorted(completed_matches, key=lambda m: m.get("utc_date") or "", reverse=True)

    upcoming_matches = [m for m in stored_matches if m.get("status") in {"SCHEDULED", "TIMED"}]
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


def compute_season_context(stored_matches: list[dict], starting_ratings: dict[str, float]) -> dict:
    finished_matches, completed_matches, upcoming_matches = split_matches_by_status(stored_matches)
    ratings, pregame_ratings = compute_elo_ratings(finished_matches, starting_ratings, include_pregame=True)

    probabilities_table = []
    upcoming_probabilities = []
    for match in upcoming_matches:
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        utc_date = match.get("utc_date")
        matchday = match.get("matchday")
        if not home_team or not away_team:
            continue
        p_home, p_draw, p_away = predict_match(home_team, away_team, ratings)
        home_elo = ratings.get(home_team, 1500.0)
        away_elo = ratings.get(away_team, 1500.0)
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
        pregame_home_elo = pregame.get("pregame_home_elo", starting_ratings.get(home_team, 1500.0))
        pregame_away_elo = pregame.get("pregame_away_elo", starting_ratings.get(away_team, 1500.0))
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

    elo_change_values = {team: rating - starting_ratings.get(team, 1500.0) for team, rating in ratings.items()}
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

st.set_page_config(page_title="Soccer Forecasting Tool", page_icon="⚽", layout="wide")

st.title("⚽ Soccer Forecasting Tool")
st.caption("Local MVP for football-data match retrieval and SQLite storage")

competition = st.selectbox(
    "Competition",
    options=["DED", "PL", "PD", "SA", "BL1", "FL1"],
    index=0,
)
season = st.number_input("Season", min_value=2000, max_value=2100, value=2025, step=1)
refresh_clicked = st.button("Refresh matches")

if refresh_clicked:
    try:
        fetched = fetch_matches(competition=competition, season=int(season))
        save_matches(fetched, competition=competition, season=int(season))
        st.success(f"Refreshed {len(fetched)} matches for {competition} {int(season)}")
    except Exception as exc:
        st.error(str(exc))

stored_matches = get_matches(competition=competition, season=int(season))

starting_ratings, starting_ratings_info = load_starting_ratings_csv()
if starting_ratings_info:
    st.info(starting_ratings_info)

season_context = compute_season_context(stored_matches, starting_ratings)
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

data_elo_tab, simulations_tab, team_deep_dive_tab = st.tabs(["Data & Elo", "Simulations", "Team Deep Dive"])

with data_elo_tab:
    st.subheader("Upcoming Match Probabilities")
    if probabilities_table:
        st.dataframe(probabilities_table, use_container_width=True)
    else:
        st.info("No upcoming matches available for probability prediction.")

    st.subheader("Completed matches")
    if completed_table:
        st.dataframe(completed_table, use_container_width=True)
    else:
        st.info("No FINISHED matches available.")

    st.subheader("Elo Ratings")
    if ratings_table:
        ratings_display_df = pd.DataFrame(ratings_table)
        ratings_display_df = ratings_display_df.drop(
            columns=[
                col
                for col in ratings_display_df.columns
                if col == "index" or str(col).startswith("Unnamed")
            ],
            errors="ignore",
        )
        ratings_display_df = ratings_display_df.reset_index(drop=True)
        ratings_height = 35 * (len(ratings_display_df) + 1) + 20
        st.dataframe(
            ratings_display_df,
            hide_index=True,
            height=ratings_height,
            use_container_width=True,
        )
    else:
        st.info("No FINISHED matches available to compute Elo ratings.")

    st.subheader("League standings")
    if standings_table:
        standings_display_df = pd.DataFrame(standings_table)
        standings_display_df = standings_display_df.drop(
            columns=[
                col
                for col in standings_display_df.columns
                if col == "index" or str(col).startswith("Unnamed")
            ],
            errors="ignore",
        )
        standings_display_df = standings_display_df.reset_index(drop=True)
        standings_height = 35 * (len(standings_display_df) + 1) + 20
        st.dataframe(
            standings_display_df,
            hide_index=True,
            height=standings_height,
            use_container_width=True,
        )
    else:
        st.info("No completed matches available to compute standings.")

with simulations_tab:
    st.header("Monte Carlo Simulation")
    num_simulations = st.number_input(
        "Number of simulations",
        min_value=100,
        max_value=20000,
        value=1000,
        step=100,
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
            key=lambda team: (-starting_ratings.get(team, 1500.0), team),
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
            .mark_rect(stroke="#d9d9d9", strokeWidth=0.6)
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
                color=alt.Color(
                    "percentage:Q",
                    title="Probability (%)",
                    scale=alt.Scale(domain=[0, 50], clamp=True, range=["#ffffff", "#1f77b4"]),
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
        probability_columns = [
            col for col in simulation_display_df.columns if col != "team"
        ]
        simulation_display_df[probability_columns] = simulation_display_df[probability_columns].round(1)
        simulation_display_df = simulation_display_df.drop(
            columns=[
                col
                for col in simulation_display_df.columns
                if col == "index" or str(col).startswith("Unnamed")
            ],
            errors="ignore",
        )
        simulation_display_df = simulation_display_df.reset_index(drop=True)
        simulation_height = 35 * (len(simulation_display_df) + 1) + 20
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
        key=lambda team: (-starting_ratings.get(team, 1500.0), team),
    )
    if not teams_for_select:
        st.info("No teams available for deep dive.")
    else:
        selected_team = st.selectbox("Team", options=teams_for_select, index=0)

        current_elo = ratings.get(selected_team, 1500.0)
        starting_elo = starting_ratings.get(selected_team, 1500.0)
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
                pregame_team_elo = starting_ratings.get(selected_team, 1500.0)
            if pregame_opponent_elo is None:
                pregame_opponent_elo = starting_ratings.get(opponent, 1500.0)
            if postgame_team_elo is None:
                postgame_team_elo = pregame_team_elo
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

        st.subheader("Completed matches")
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
        for match in team_upcoming_matches:
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            is_home = home_team == selected_team
            opponent = away_team if is_home else home_team
            home_away = "Home" if is_home else "Away"
            team_elo = ratings.get(selected_team, 1500.0)
            opponent_elo = ratings.get(opponent, 1500.0)
            p_home, p_draw, p_away = predict_match(home_team, away_team, ratings)
            p_win = p_home if is_home else p_away
            p_loss = p_away if is_home else p_home
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

        st.subheader("Remaining fixtures")
        if remaining_rows:
            remaining_df = pd.DataFrame(remaining_rows)
            remaining_column_config = {
                "p_win": st.column_config.NumberColumn(format="%.1f"),
                "p_draw": st.column_config.NumberColumn(format="%.1f"),
                "p_loss": st.column_config.NumberColumn(format="%.1f"),
            }
            st.dataframe(
                remaining_df,
                use_container_width=True,
                column_config=remaining_column_config,
            )
        else:
            st.info("No remaining fixtures for selected team.")

        simulation_matrix = st.session_state.get("simulation_position_matrix")
        if simulation_matrix is not None and isinstance(simulation_matrix, pd.DataFrame) and not simulation_matrix.empty:
            team_sim = simulation_matrix[simulation_matrix["team"] == selected_team]
            if not team_sim.empty:
                team_sim = team_sim.iloc[0].to_dict()
                team_sim_rows = []
                for col, value in team_sim.items():
                    if col == "team":
                        continue
                    team_sim_rows.append({"position": col, "probability": round(float(value), 1)})
                team_sim_rows = sorted(team_sim_rows, key=lambda r: int(r["position"]))
                st.subheader("Finishing-position probabilities")
                st.dataframe(team_sim_rows, use_container_width=True)
