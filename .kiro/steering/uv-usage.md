---
inclusion: always
---

# Use uv for all Python commands

gsolve **is** a uv-managed project — do not re-check. If `uv` is not on PATH, STOP and advise installing it (<https://docs.astral.sh/uv/getting-started/installation/>); do not fall back to a global interpreter.

## `uv sync --all-groups`
- Run before anything else — syncs all dependency groups.
- `uv sync --all-groups --locked` reproduces CI exactly.

## `uv run <tool>`
- Run every Python tool through it: `uv run pytest`, `uv run ruff check`, `uv run python ...`.
- Never a bare `python` / `pytest` / `ruff`.
- Never `pip install` — use `uv add` / `uv add --group <group>`.

## Missing tool → STOP
- If `uv run <tool>` fails because the tool is missing (not installed / no such command), STOP — do not fall back to a global or ad-hoc install.
- Alert the user that the tool is not in the project environment and advise adding it to a dependency group: `uv add --group dev <tool>` (or another appropriate group).

## Interpreter outside `.venv/`
- If `uv run python -c "import sys; print(sys.executable)"` resolves outside `.venv/`, that is a deliberately configured interpreter (`UV_PROJECT_ENVIRONMENT` or a pinned interpreter) — note it, don't override.
