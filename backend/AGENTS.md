# Repository Guidelines

## Project Structure & Module Organization
- Source: `backend/` package with domain modules:
  - `auth/` (JWT, users), `routes/` (FastAPI routers), `crud/` (DB helpers),
    `super_models/` (SQLAlchemy models), `db/` (SQLite data + schema templates),
    `utils/` (graph, inference, embedding), `graph_base/` (graph data).
- App entrypoint: `backend/main.py` (creates FastAPI app, mounts routers).
- Database: SQLite at `backend/db/data_tables/intentional.db` (auto-created).
- Example script: `backend/test_openai_to_graph.py` builds a mock graph.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r backend/requirements.txt`
- Run API (from repo root): `uvicorn backend.main:app --reload --port 8000`
  - Interactive docs: http://localhost:8000/docs
- Quick graph test: `python backend/test_openai_to_graph.py`

## Coding Style & Naming Conventions
- Python 3.11+, PEP 8, 4-space indentation, UTF-8.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE`.
- Types: prefer standard annotations (e.g., `list[dict]`).
- Imports: use absolute package imports (e.g., `from backend.utils ...`).
- Lint/format: no enforced tool; match existing style. If used locally, prefer Black (88 cols) and Ruff.

## Testing Guidelines
- Framework: lightweight scripts today (see `backend/test_openai_to_graph.py`).
- Convention: add Pytest tests under `backend/tests/`, files named `test_*.py`.
- Run all tests (if added): `pytest -q`.
- Aim for coverage of routes (via TestClient), CRUD functions, and model serialization.

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (<= 72 chars). Example: `Add analyzer route for deep analysis`.
- Scope: group related changes; include rationale in body when non-trivial.
- PRs must include:
  - Problem statement and summary of changes.
  - Steps to reproduce/verify (commands like `uvicorn backend.main:app ...`).
  - Linked issues (e.g., `Closes #123`) and screenshots or sample payloads for new routes.

## Security & Configuration Tips
- Secrets: store in `.env`; do not commit. Required keys commonly include `OPENAI_API_KEY`.
- JWT: `auth/jwt_handler.py` uses a placeholder secret; replace with an env var in deployment.
- CORS: `main.py` allows `http://localhost:5173`; adjust per environment.
