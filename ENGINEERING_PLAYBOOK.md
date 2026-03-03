# Engineering Playbook

Short, reusable patterns proven in this repo. Keep this file compact and practical.

## Streamlit

### Cache-key correctness
- Trigger phrase: `cache key`
- Rule: cache derived views/rankings using a deterministic content signature, not freshness proxies.
- Pitfall: `row_count + max_date` can miss in-place row corrections.
- Seen in: `app.py` on 2026-03-02 (stale cache bug).

### File config invalidation
- Trigger phrase: `mtime`
- Rule: cache file-based config/seed loaders with `(path, mtime)` to auto-invalidate on edits.
- Pitfall: path-only cache keys can serve stale config.
- Seen in: `app.py` on 2026-03-02 (CSV parse hot path).

### Cached mutability safety
- Trigger phrase: `mutable cached object`
- Rule: do not mutate cached return objects in-place; copy before transformations.
- Pitfall: cross-rerun state bleed from mutated cache objects.

### Fallback observability
- Trigger phrase: `logger.exception`
- Rule: keep user fallback behavior stable, add contextual exception logs for root cause.
- Pitfall: silent failures with no diagnostics.
- Seen in: `app.py` on 2026-03-02 (generic fallback visibility gap).

### Deprecation migrations
- Trigger phrase: `deprecation migration`
- Rule: do strict mechanical migration early when deprecations appear.
- Pitfall: delayed migration increases risk near removal windows.
- Seen in: `app.py` on 2026-03-02 (`use_container_width` -> `width`).

## SQLite

### Lock contention hardening
- Trigger phrase: `busy_timeout`
- Rule: set connect timeout and `PRAGMA busy_timeout` for rerun-heavy apps.
- Pitfall: transient `database is locked` errors under concurrent reruns.
- Seen in: `app.py` on 2026-03-02 (operational hardening).

### Query-shape indexing
- Trigger phrase: `composite index`
- Rule: index exact dominant `WHERE` + `ORDER BY` patterns.
- Pitfall: generic/missing indexes degrade read performance as data grows.
- Seen in: `app.py` on 2026-03-02 (matches query pattern).

### Write amplification guard
- Trigger phrase: `skip unchanged writes`
- Rule: compare fetched vs DB content signatures before writing.
- Pitfall: unconditional refresh writes increase lock pressure and churn.
- Seen in: `app.py` on 2026-03-02 (DB write churn).

### Single DB path source
- Trigger phrase: `DB_PATH single source`
- Rule: define one `DB_PATH` constant and import it everywhere.
- Pitfall: duplicated path literals can split reads/writes across files.
- Seen in: `app.py` on 2026-03-02 (path unification).
