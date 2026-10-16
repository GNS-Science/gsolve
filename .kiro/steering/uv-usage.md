---
inclusion: always
---

# Use uv for all Python commands

Applies to any **uv-managed** project (one with both `uv.lock` and
`pyproject.toml`).

1. Sync all dependency groups before running anything: `uv sync --all-groups`
   (use `uv sync --all-groups --locked` to reproduce CI exactly).
2. Run every Python tool through `uv run`: `uv run pytest`, `uv run ruff check`,
   `uv run python ...`. Never a bare `python`/`pytest`/`ruff`, never
   `pip install` (use `uv add` / `uv add --group <group>`).
3. If `uv run python -c "import sys; print(sys.executable)"` resolves outside
   `.venv/`, that reflects a deliberately configured interpreter
   (`UV_PROJECT_ENVIRONMENT` or a pinned interpreter) — note it, don't override.
