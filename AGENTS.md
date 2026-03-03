# AGENTS.md

Project guidance for coding agents working in this repository.

## 1) Scope and Architecture
- Keep this as a single-process, monolithic Streamlit app.
- No new service layers, no package restructuring, no new dependencies unless explicitly requested.
- Prefer minimal, localized edits over broad refactors.

## 2) File Ownership
- `app.py`: Streamlit UI, orchestration, caching flow, diagnostics, simulation, tab layouts.
- `elo.py`: Elo update logic and 3-way probability mapping only.
- `db.py`: SQLite schema + persistence/query helpers.
- `data_client.py`: football-data.org API fetch + normalization.
- `starting_elo.csv`: competition-scoped seed Elo values.
- `model_config.csv`: competition-scoped home advantage values.

## 3) Modeling Invariants
Do not change these unless explicitly requested:
- Elo update framework and K-factor behavior.
- Goal-difference multiplier behavior.
- Monte Carlo simulation structure (fixed precomputed probabilities, no in-sim Elo updates).
- Diagnostics definitions (log loss/Brier/calibration/matchday metrics) and baseline meaning.

Any model changes must be restricted to the requested function(s), usually `elo.py`.

## 4) Data and Caching Semantics
- SQLite (`soccer.db`) is the local persistent store used by app logic.
- API fetches use Streamlit cache semantics (`st.cache_data`) and are instance-local on Streamlit Cloud.
- In aggregate diagnostics mode, multi-league data may still depend on per-instance cache/DB state.
- Do not change DB schema unless explicitly requested.

## 5) UI Change Rules
- Keep existing table column order, naming, and formatting unless explicitly asked to change.
- Avoid changing behavior across tabs when request scope is tab-specific.
- For chart fixes, prefer minimal visual changes without touching underlying computations.

## 6) Editing Discipline
- Make only the requested change scope.
- Do not opportunistically refactor unrelated code.
- Preserve existing behavior/outputs unless the request explicitly requires behavior changes.
- If a requested fix can be done in one file, do it in one file.

## 7) Validation Checklist
After edits:
1. Run syntax check on changed files (at minimum):
   - `python3 -m py_compile app.py elo.py db.py data_client.py`
   - or only changed files if scope is narrow.
2. Summarize:
   - files changed
   - what was changed
   - what was intentionally left unchanged

## 8) Current Product Constraints
- Season support is currently fixed to `2025`.
- Supported competitions are controlled in `COMPETITION_OPTIONS` in `app.py`.
- Global Ratings tab uses local DB snapshot and computes Elo on-demand across supported leagues.

## 9) Communication Style
- Be concise and technical.
- Report mismatches/errors directly.
- If a request may have side effects beyond scope, state it before editing.

## 10) Agent Audit Checklist
Use this only for periodic quality sweeps or when explicitly requested.

- Cache correctness: any derived/cache-heavy path keyed by freshness proxies instead of content signature?
- Write churn: any unconditional `save_matches(...)`/DB refresh writes that should be content-gated?
- Error visibility: any broad `except Exception` fallback without contextual logging?
- SQLite resilience: any new `sqlite3.connect(...)` call missing timeout/busy-timeout handling?
- Config cache invalidation: any file-backed loaders missing mtime-aware cache keys?
- Path consistency: any duplicated DB path literals instead of shared `DB_PATH`?
- Deprecation hygiene: any reintroduced `use_container_width` usages?
- Scope guard: propose only high-confidence, low-risk fixes; no architecture refactor in audit mode.
