# CLAUDE.md

gsolve is **uv-managed** (`uv.lock` + `pyproject.toml` with `[dependency-groups]`). Always:

1. `uv sync --all-groups` before running anything (installs the `dev` and `docs`
   groups). Use `uv sync --all-groups --locked` to match CI exactly.
2. Run every Python tool through `uv run` — `uv run pytest`, `uv run ruff check src/ tests/`,
   `uv run python ...`. Never a bare `python`/`pytest`/`ruff`, never `pip install`
   (use `uv add` / `uv add --group dev`).
3. Lint suppression uses `# ruff: ignore`, never `# noqa`.
