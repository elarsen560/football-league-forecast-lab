# Migration Notes

This repo is intended to be copied from its current Desktop location to:

```bash
~/projects/soccer
```

Copy the full directory structure, including local data/config files such as:

- `soccer.db`
- `starting_elo.csv`
- `model_config.csv`
- `AGENTS.md`
- `ENGINEERING_PLAYBOOK.md`
- `docs/`

## Runtime Path Assumptions

The app uses relative paths for local runtime files:

- SQLite DB: `soccer.db`
- Starting Elo CSV: `starting_elo.csv`
- Model config CSV: `model_config.csv`

Run commands from the repo root after migration:

```bash
cd ~/projects/soccer
streamlit run app.py
```

Do not run `streamlit run /absolute/path/to/app.py` from another working directory unless DB/config paths are adjusted.

## One-Time Setup After Copy

Recreate the virtual environment in the new location rather than relying on the copied `.venv`:

```bash
cd ~/projects/soccer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ensure the API key is available in the shell or Streamlit Cloud secrets:

```bash
export FOOTBALL_DATA_API_KEY="your_key_here"
```

## Future Codex Notes

- Treat `~/projects/soccer` as the active repo after migration.
- Avoid editing the old Desktop copy once the migration is complete.
- If both copies exist, their `soccer.db` files can diverge after data refreshes.
- Streamlit cache state is not portable and will rebuild naturally.
