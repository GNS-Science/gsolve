---
inclusion: always
---

# Use uv for all Python commands

gsolve is **uv-managed** (`uv.lock` + `pyproject.toml` with `[dependency-groups]`).

1. Sync all groups before running anything: `uv sync --all-groups`
   (use `uv sync --all-groups --locked` to reproduce CI exactly).
2. Run every Python tool through `uv run`: `uv run pytest`,
   `uv run ruff check src/ tests/`, `uv run python ...`. Never a bare
   `python`/`pytest`/`ruff`, never `pip install` (use `uv add` / `uv add --group dev`).
3. Lint suppression uses `# ruff: ignore`, never `# noqa`.
