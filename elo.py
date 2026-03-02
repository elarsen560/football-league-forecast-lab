import math

DEFAULT_ELO = 1800.0
DEFAULT_HOME_ADVANTAGE = 75.0


def compute_elo_ratings(
    matches: list[dict],
    starting_ratings: dict[str, float] = None,
    include_pregame: bool = False,
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> dict[str, float] | tuple[dict[str, float], dict[int, dict[str, float]]]:
    starting_rating = DEFAULT_ELO
    k_factor = 20.0
    ratings: dict[str, float] = {}
    pregame_ratings: dict[int, dict[str, float]] = {}
    starting_ratings = starting_ratings or {}

    for match in matches:
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        home_score = match.get("home_score")
        away_score = match.get("away_score")

        if not home_team or not away_team:
            continue
        if home_score is None or away_score is None:
            continue

        home_rating = ratings.get(home_team, starting_ratings.get(home_team, starting_rating))
        away_rating = ratings.get(away_team, starting_ratings.get(away_team, starting_rating))
        match_id = match.get("match_id")
        if include_pregame and match_id is not None:
            pregame_ratings[match_id] = {
                "pregame_home_elo": home_rating,
                "pregame_away_elo": away_rating,
            }

        expected_home = 1.0 / (1.0 + 10.0 ** ((away_rating - (home_rating + home_advantage)) / 400.0))
        expected_away = 1.0 - expected_home

        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score < away_score:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5

        goal_difference = home_score - away_score
        multiplier = max(1.0, math.sqrt(abs(goal_difference)))
        ratings[home_team] = home_rating + k_factor * multiplier * (actual_home - expected_home)
        ratings[away_team] = away_rating + k_factor * multiplier * (actual_away - expected_away)
        if include_pregame and match_id is not None:
            pregame_ratings[match_id]["postgame_home_elo"] = ratings[home_team]
            pregame_ratings[match_id]["postgame_away_elo"] = ratings[away_team]

    if include_pregame:
        return ratings, pregame_ratings
    return ratings


def predict_match(
    home_team: str,
    away_team: str,
    ratings: dict[str, float],
    home_advantage: float = DEFAULT_HOME_ADVANTAGE,
) -> tuple[float, float, float]:
    """
    Return (p_home, p_draw, p_away) using current Elo ratings.
    Method:
      - Apply configurable home advantage to home Elo before expected-score calculation.
      - Compute base expected probabilities E_home and E_away with Elo formula.
      - Compute dynamic draw probability from E_home * E_away.
      - Split draw mass equally from home/away expectations.
    """
    starting_rating = DEFAULT_ELO
    r_home = ratings.get(home_team, starting_rating)
    r_away = ratings.get(away_team, starting_rating)
    r_home_with_adv = r_home + home_advantage

    e_home = 1.0 / (1.0 + 10.0 ** ((r_away - r_home_with_adv) / 400.0))
    e_away = 1.0 - e_home
    p_draw_raw = (4.0 / 3.0) * (e_home * e_away)

    a = 0.05
    p_draw = math.sqrt((a * a) + (p_draw_raw * p_draw_raw))
    denom = math.sqrt((a * a) + ((1.0 / 3.0) * (1.0 / 3.0)))
    p_draw = (p_draw / denom) * (1.0 / 3.0)

    b = 3.0 * ((1.0 / 3.0) - p_draw_raw)
    b = max(0.0, min(1.0, b))
    f = 0.5 + (0.5 * b)

    if e_home > e_away:
        p_home = e_home - (f * p_draw)
        p_away = e_away - ((1.0 - f) * p_draw)
        if p_away < 0.0:
            p_away = 0.0
            p_home = 1.0 - p_draw
    elif e_away > e_home:
        p_home = e_home - ((1.0 - f) * p_draw)
        p_away = e_away - (f * p_draw)
        if p_home < 0.0:
            p_home = 0.0
            p_away = 1.0 - p_draw
    else:
        p_home = e_home - (0.5 * p_draw)
        p_away = e_away - (0.5 * p_draw)

    p_home = max(0.0, p_home)
    p_away = max(0.0, p_away)
    p_draw = max(0.0, p_draw)
    s = p_home + p_draw + p_away
    if s > 0.0 and abs(s - 1.0) > 1e-9:
        p_home /= s
        p_draw /= s
        p_away /= s
    return p_home, p_draw, p_away
