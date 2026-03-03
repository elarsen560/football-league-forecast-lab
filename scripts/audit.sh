#!/usr/bin/env bash
set -euo pipefail

echo "[audit] Repo scan: broad exception handlers"
rg -n "except Exception" app.py db.py data_client.py elo.py || true

echo
echo "[audit] Repo scan: deprecated Streamlit width flag"
rg -n "use_container_width" app.py db.py data_client.py elo.py || true

echo
echo "[audit] Repo scan: sqlite connect points"
rg -n "sqlite3\\.connect\\(" db.py app.py || true

echo
echo "[audit] Syntax check"
python3 -m py_compile app.py elo.py db.py data_client.py

echo
echo "[audit] Done"
