# CLAUDE.md

gsolve **is** a uv-managed project — do not re-check. If `uv` is not on PATH, STOP and advise installing it (<https://docs.astral.sh/uv/getting-started/installation/>); do not fall back to a global interpreter. Always:

1. `uv sync --all-groups` before running anything (installs the `dev` and `docs`
   groups). Use `uv sync --all-groups --locked` to match CI exactly.
2. Run every Python tool through `uv run` — `uv run pytest`, `uv run ruff check src/ tests/`,
   `uv run python ...`. Never a bare `python`/`pytest`/`ruff`, never `pip install`
   (use `uv add` / `uv add --group dev`).
3. If `uv run <tool>` fails because the tool is missing (not installed / no such
   command), STOP — don't fall back to a global or ad-hoc install. Alert the user
   and advise adding it to a dependency group: `uv add --group dev <tool>` (or
   another appropriate group).
4. Lint suppression uses `# ruff: ignore`, never `# noqa`.
