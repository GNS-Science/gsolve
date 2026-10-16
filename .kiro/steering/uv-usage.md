---
inclusion: always
---

# Use uv for all Python commands

gsolve **is** a uv-managed project — do not re-check. If `uv` is not on PATH,
STOP and advise installing it (<https://docs.astral.sh/uv/getting-started/installation/>);
do not fall back to a global interpreter.

1. Sync all dependency groups before running anything: `uv sync --all-groups`
   (use `uv sync --all-groups --locked` to reproduce CI exactly).
2. Run every Python tool through `uv run`: `uv run pytest`, `uv run ruff check`,
   `uv run python ...`. Never a bare `python`/`pytest`/`ruff`, never
   `pip install` (use `uv add` / `uv add --group <group>`).
3. If `uv run <tool>` fails because the tool is missing (not installed / no such
   command), STOP — do not fall back to a global or ad-hoc install. Alert the
   user that the tool is not in the project environment and advise adding it to a
   dependency group: `uv add --group dev <tool>` (or another appropriate group).
4. If `uv run python -c "import sys; print(sys.executable)"` resolves outside
   `.venv/`, that reflects a deliberately configured interpreter
   (`UV_PROJECT_ENVIRONMENT` or a pinned interpreter) — note it, don't override.
