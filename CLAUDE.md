# CLAUDE.md

gsolve **is** a uv-managed project — do not re-check. If `uv` is not on PATH, STOP and advise installing it (<https://docs.astral.sh/uv/getting-started/installation/>); do not fall back to a global interpreter.

## `uv sync --all-groups`
- Run before anything else — installs the `dev` and `docs` groups.
- `uv sync --all-groups --locked` matches CI exactly.

## `uv run <tool>`
- Run every Python tool through it: `uv run pytest`, `uv run ruff check src/ tests/`, `uv run python ...`.
- Never a bare `python` / `pytest` / `ruff`.
- Never `pip install` — use `uv add` / `uv add --group dev`.

## Missing tool → STOP
- If `uv run <tool>` fails because the tool is missing (not installed / no such command), STOP — do not fall back to a global or ad-hoc install.
- Alert the user and advise adding it to a dependency group: `uv add --group dev <tool>` (or another appropriate group).

## Lint suppression
- Use `# ruff: ignore`, never `# noqa`.
