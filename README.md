# Football League Forecast Lab

Streamlit app for league forecasting using football-data.org match data, SQLite storage, Elo-based probabilities, Monte Carlo season simulation, and diagnostics.

## What the app does

- Pulls match data from football-data.org for selected competition/season.
- Stores matches locally in SQLite (`soccer.db`).
- Builds Elo ratings from completed matches.
- Predicts upcoming match probabilities (home/draw/away).
- Runs Monte Carlo season simulations from current state.
- Shows team- and rank-level uncertainty (entropy) from simulation outputs.
- Provides diagnostics (calibration, log loss, Brier score, matchday performance).
- Includes Team Deep Dive with Elo evolution and fixture-level probability context.

## Current scope

- **Season support:** 2025 only.
- **Competition support:**
  - Eredivisie (`DED`)
  - Premier League (`PL`)
  - Championship (`ELC`)
  - Bundesliga (`BL1`)
  - Serie A (`SA`)
  - La Liga (`PD`)
  - Ligue 1 (`FL1`)
  - Primeira Liga (`PPL`)

## Project structure

- `app.py`  
  Main Streamlit app: UI, data loading, caching flow, season context construction, simulation, diagnostics, entropy visualizations.
- `data_client.py`  
  football-data.org client and response normalization.
- `db.py`  
  SQLite schema and CRUD helpers for match storage.
- `elo.py`  
  Elo rating updates and match probability model.
- `starting_elo.csv`  
  Seed Elo ratings by competition/team.
- `model_config.csv`  
  League-specific home-advantage config.
- `soccer.db`  
  Local SQLite database file (created automatically).

## Tech stack

- Frontend/runtime: **Streamlit**
- Data wrangling: **pandas**
- Storage: **SQLite**
- External API: **football-data.org v4**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set API key locally:

```bash
export FOOTBALL_DATA_API_KEY="your_key_here"
```

For Streamlit Cloud, set `FOOTBALL_DATA_API_KEY` in app secrets/environment.

## Run

```bash
streamlit run app.py
```

## Configuration files

### `starting_elo.csv`

Required columns:

- `competition`
- `team`
- `rating`

Example:

```csv
competition,team,rating
PL,Arsenal FC,1835
DED,PSV,1780
```

Behavior:
- Ratings are filtered by selected competition.
- Missing teams fall back to default Elo (`1800`).
- Missing/malformed file falls back to defaults with Streamlit info message.

### `model_config.csv`

Required columns:

- `competition`
- `home_advantage`

Example:

```csv
competition,home_advantage
PL,35
DED,85
BL1,35
SA,50
PD,100
```

Behavior:
- Used to set league-specific home advantage.
- Missing/malformed file falls back to default (`75`).
- Config load is cached with file mtime-aware invalidation.

## Data refresh & caching behavior

- API fetches use `@st.cache_data(ttl=3600)` by `(competition, season)`.
- Selected competition is refreshed in normal page flow (hourly cache semantics).
- In Diagnostics with **Aggregate across all leagues** checked:
  - app attempts cached refresh for each supported league,
  - saves successful responses to DB,
  - falls back to local DB per league if refresh fails (with warning).

Notes for Streamlit Cloud:
- Cache and local DB are instance-local, not globally shared across all users/instances.
- A fresh/restarted instance may need to repopulate leagues.

## Data model (SQLite)

`matches` table:

- `match_id INTEGER PRIMARY KEY`
- `competition TEXT NOT NULL`
- `season INTEGER NOT NULL`
- `utc_date TEXT`
- `status TEXT`
- `matchday INTEGER`
- `home_team TEXT`
- `away_team TEXT`
- `home_score INTEGER`
- `away_score INTEGER`

Storage/update pattern:
- `save_matches(...)` currently does `DELETE` by `(competition, season)` then `INSERT OR REPLACE` for fetched rows.

## Modeling summary

### Elo update (`elo.py`)

- Starting Elo default: `1800`
- K-factor: `20`
- Home advantage default: `75` (overrideable per league via config)
- Expected score uses standard Elo logistic with 400-point scale.
- Match result mapping: win `1.0`, draw `0.5`, loss `0.0`.
- Goal-difference multiplier: `max(1, sqrt(abs(goal_diff)))`.
- Supports pregame/postgame snapshot output for completed matches.

### Match probability model (`predict_match`)

- Base home expectation from Elo (with configured home advantage).
- Draw model (dynamic):
  - `p_draw = (4/3) * (E_home * E_away)`
- Win/loss adjusted by splitting draw mass:
  - `p_home = E_home - p_draw/2`
  - `p_away = E_away - p_draw/2`

## Simulation summary

- Monte Carlo simulations use fixed upcoming match probabilities (no Elo updates in-sim).
- Start from current points table.
- Simulated outcomes award points (3/1/0).
- Final ordering tiebreakers:
  1. points
  2. current goal difference
  3. current goals for
  4. team name
- Outputs:
  - position-probability heatmap
  - position-probability matrix table
  - Team Entropy
  - Rank Entropy

## Diagnostics summary

Diagnostics tab supports selected league or pooled mode.

Includes:
- Season-to-date metrics:
  - model log loss / Brier
  - uniform baseline
  - prevalence baseline
  - expected (model-implied difficulty) comparator
- Calibration curves with Wilson 95% intervals:
  - home win / draw / home loss
- Delta-binned calibration (pregame Elo + home advantage)
- Matchday performance charts:
  - log loss and Brier series for Model / Uniform / Prevalence / Expected
  - beat-rate summaries for `n >= 5` matchdays

## Team Deep Dive summary

For selected team:
- Current Elo, Elo delta, league rank/points
- Elo evolution chart across completed matches
- Completed match table with pregame Elo context and per-match Elo change
- Remaining fixtures with team/opponent Elo and win/draw/loss probabilities

## Known limitations

- Single-process monolithic Streamlit app (no background jobs, no task queue).
- Local SQLite only; no shared persistent database across cloud instances.
- Season fixed to 2025.
- API rate limits and upstream status timing can affect freshness.

## Future improvements

- **Reduce unnecessary DB writes on cached fetches**:
  skip `save_matches(...)` when fetch result is from unchanged cache to reduce write churn.
- Persist per-league refresh metadata in DB and refresh only stale leagues explicitly.
- Add optional simulation mode with in-simulation Elo updates.
- Add tests for data loading, Elo math, and diagnostics aggregations.
- Improve observability around per-league refresh status and cache hits.
