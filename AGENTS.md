---
inclusion: always
---

# gsolve agent instructions

Single source of truth for all AI coding tools. Kiro loads this via the `.kiro/steering/AGENTS.md` symlink; the `inclusion: always` frontmatter above is for Kiro and is ignored by other tools.

## Use uv for all Python commands

gsolve **is** a uv-managed project — do not re-check. If `uv` is not on PATH, STOP and advise installing it (<https://docs.astral.sh/uv/getting-started/installation/>); do not fall back to a global interpreter.

### `uv sync --all-groups`

- Run before anything else — syncs all dependency groups.
- `uv sync --all-groups --locked` reproduces CI exactly.

### `uv run <tool>`

- Run every Python tool through it: `uv run pytest`, `uv run ruff check src/ tests/`, `uv run python ...`.
- Never a bare `python` / `pytest` / `ruff`.
- Never `pip install` — use `uv add` / `uv add --group <group>`.

### Missing tool → STOP

- If `uv run <tool>` fails because the tool is missing (not installed / no such command), STOP — do not fall back to a global or ad-hoc install.
- Alert the user that the tool is not in the project environment and advise adding it to a dependency group: `uv add --group dev <tool>` (or another appropriate group).

### Interpreter outside `.venv/`

- If `uv run python -c "import sys; print(sys.executable)"` resolves outside `.venv/`, that is a deliberately configured interpreter (`UV_PROJECT_ENVIRONMENT` or a pinned interpreter) — note it, don't override.

## Lint suppression

- Use `# ruff: ignore`, never `# noqa`.

## External-service auth errors → STOP

On an authentication/authorization failure from any external service (GitHub, AWS, GitLab, npm, Docker registry, any credentialed API) — 401/403, expired/invalid token, SSO or login required, bad credentials — STOP immediately.

### Do

- Report the exact error.
- Tell the user to log in / refresh the token (e.g. `aws sso login`, `gh auth login`), then wait for them.

### Don't

- Retry the call, run tests, edit config, switch profiles/endpoints, or otherwise thrash to work around it.
- Auth is the user's to fix — resume only after they confirm they've re-authenticated.
