## `dnadesign` for agents

This repository is a uv-managed Python monorepo containing modular tools under `src/dnadesign/`.
Treat this file as a navigation map; canonical policy and operational detail live in system-of-record docs and runbooks linked below.

## Start here (system of record)

- Architecture map: `ARCHITECTURE.md`
- Engineering invariants: `DESIGN.md`
- Security/secrets policy: `SECURITY.md`
- Reliability and operational model: `RELIABILITY.md`
- Planning lifecycle: `PLANS.md`
- Quality score scaffold: `QUALITY_SCORE.md`
- Full docs index: `docs/README.md`

## Repo map

- Root docs: `README.md`, `docs/`
- Python code: `src/dnadesign/`
- CI: `.github/workflows/ci.yaml`
- Tooling: `.pre-commit-config.yaml`, `pyproject.toml`, `uv.lock`, `pixi.lock`
- Local tool instructions: follow nearest `AGENTS.md` inside each tool directory.

## Task map

- Setup environment:
  - `uv sync --locked`
  - `uv sync --locked --group dev`
- Validate changes:
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run pytest -q`
  - `uv run python -m dnadesign.devtools.docs_checks`
- Discover CLIs:
  - check `[project.scripts]` in `pyproject.toml`
  - run `uv run <script> --help`

## Operational references

- BU SCC operations source of truth: `docs/bu-scc/sge-hpc-ops/SKILL.md`
- BU SCC quickstart: `docs/bu-scc/quickstart.md`
- BU SCC batch + notify runbook: `docs/bu-scc/batch-notify.md`
- Submit-ready BU SCC jobs: `docs/bu-scc/jobs/README.md`
- Notify operator docs: `docs/notify/README.md`
- Notify event contract: `docs/notify/usr-events.md`
- Maintainer docs and CI/testing notes: `docs/dev/README.md`

## Repository patterns

- Many tools are direct package layouts under `src/dnadesign/<tool>/`.
- Some tools are mini-projects with nested `src/` (for example `permuter`, `opal`) and local configs/jobs/notebooks.
- Prefer explicit tool boundaries and artifact contracts over cross-tool internal coupling.

## Generated artifacts policy

Treat these paths as generated/run artifacts unless local tool docs explicitly state otherwise:

- `**/outputs/**`
- `**/batch_results/**`
- `**/runs/**`
- `**/.pytest_cache/**`
- `src/dnadesign/**/campaigns/**/outputs/**`

Rules:

- Do not hand-edit generated outputs; fix code/config and regenerate.
- Ask before committing generated artifacts or large binaries.
- Avoid changes under `src/dnadesign/archived/` and `src/dnadesign/prototypes/` unless explicitly requested.

## Definition of done

- Diff reviewed
- Tests pass or scope-limited rationale documented
- Ruff checks pass
- Docs checks pass
- No generated artifacts committed unintentionally
- Changes are on a branch and ready for PR review
