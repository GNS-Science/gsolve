---
inclusion: always
---

## Hello LLMs and Devs

This is the single source of truth for all AI coding tools and developers. Kiro loads this via the `.kiro/steering/AGENTS.md` symlink; the `inclusion: always` frontmatter above is for Kiro and is ignored by other tools.

## Project Context

This is a Python project built with pandas, pydantic, xarray.

## Environment & Dependency Management

This project uses `uv` with environment in `.venv/`. Never use standard pip or any other tool.
The guiding principle is to maintain consistent and repeatable environment and tooling.

- **Package Manager:** `uv`
- **Environment Setup:** `uv sync --all-groups --locked` to install dependencies. Run at:
  - start of each session
  - before committing
  - after adding or removing dependencies
- **Adding Packages:** `uv add <package>` (Never modify pyproject.toml manually)
- **Removing Packages:** `uv remove <package>` (Never modify pyproject.toml manually)
- **Lockfile:** `uv lock`
- **Python** `uv run python` (never a bare `python`)
  - **Python outside venv:** If `uv run python -c "import sys; print(sys.executable)"` resolves outside `.venv/`, that is a deliberately configured interpreter (`UV_PROJECT_ENVIRONMENT` or a pinned interpreter) — note it, don't override.

## Tooling

### Linting and Testing tools

Must be installed as project dependencies and run via `uv run <tool>`.
Never install/use tools globally or ad-hoc.

- Install via `uv add --group dev <tool>`
- Invoked via `uv run <tool>`

### Non standard tools

Run via `uvx <tool>` (e.g. `uvx prek ...`).

### Ambiguous tools

Ask user for clarification.

## Quality Gates & Style Guidelines

Every file created or modified must pass these exact checks before you signal completion:

- **Linter & Formatter:** `uv run ruff check --fix` and `uv run ruff format`
- **Lint suppression** Use `# ruff: ignore`, never `# noqa`.
- **Type Checker:** `uv run ty check`
- **Type error suppression** Use `# ty: ignore`.

## Testing Framework

You must verify your work by writing and/or running tests. Do not write throwaway scripts to test features.

- **Test Runner:** `uv run pytest`
- **Run Specific Test:** `uv run pytest tests/test_filename.py -k "test_function_name"`
- **Coverage Check:** `uv run pytest --cov`

## One Shot Test and Lint All

Pre-commit hooks are configured using `prek`. To manual run for all tests and files:

- `uvx prek run -d src/ -d tests/ -d examples/ -d docs/`

## External-service auth errors → STOP

On an authentication/authorization failure from any external service (GitHub, AWS, GitLab, npm, Docker registry, any credentialed API) — 401/403, expired/invalid token, SSO or login required, bad credentials — STOP immediately.

### Do

- Report the exact error.
- Tell the user to log in / refresh the token (e.g. `aws sso login`, `gh auth login`), then wait for them.

### Don't

- Retry the call, run tests, edit config, switch profiles/endpoints, or otherwise thrash to work around it.
- Auth is the user's to fix — resume only after they confirm they've re-authenticated.
