# SECURITY

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-19

## At a glance
This document records security expectations for code, data, secrets, and dependency handling in `dnadesign`.
It is a policy map with links to operator runbooks and implementation details.

## Contents
- [Secrets and credentials](#secrets-and-credentials)
- [Dependency and supply-chain controls](#dependency-and-supply-chain-controls)
- [Data handling expectations](#data-handling-expectations)
- [Incident/reporting workflow](#incidentreporting-workflow)
- [Enforcement controls](#enforcement-controls)
- [References](#references)

## Secrets and credentials
- Never commit credentials or webhook URLs to git-tracked files.
- Prefer environment variables and secret backends for runtime secrets.
- Notify webhook and profile operations must keep secret material out of logs/history where possible.

## Dependency and supply-chain controls
- Python dependencies are pinned via `uv.lock` and installed with `uv sync --locked`.
- Pixi-managed external tools are pinned with `pixi.lock` and installed with `pixi install --locked`.
- CI and local workflows should avoid unpinned installs for operational paths.

## Data handling expectations
- Treat dataset and run artifacts as operational data, not source-of-truth code.
- Do not hand-edit generated outputs; regenerate from code/config.
- Validate external sync/remote operations explicitly; do not assume side effects succeeded.

## Incident/reporting workflow
- If a secret is discovered in repo history or working tree, stop and notify maintainers immediately.
- Rotate compromised tokens/webhooks first, then remediate repository history and runtime configs.
- Capture corrective actions in maintainer notes (`docs/dev/journal.md`) and relevant runbooks.

## Enforcement controls
- Pre-commit hooks enforce secret hygiene and key detection (`.pre-commit-config.yaml`):
  - `detect-private-key`
  - `detect-secrets` with `.secrets.baseline`
- CI enforces secrets hygiene as a blocking lane (`.github/workflows/ci.yaml`):
  - `dnadesign.devtools.secrets_baseline_check` verifies `.secrets.baseline` paths still exist in the repo tree.
  - `pre-commit run detect-secrets --all-files` scans the full tracked tree against baseline policy.
- Core CI lane runs pre-commit checks on PR diff or full tree (`.github/workflows/ci.yaml`).
- CI validates workflow definitions using `check-github-workflows` via pre-commit configuration.

## References
- Root agent map and safety rules: `AGENTS.md`
- Notify operator manual: `docs/notify/README.md`
- USR events/operator contracts: `docs/notify/usr-events.md`
- BU SCC install and ops docs: `docs/bu-scc/install.md`, `docs/bu-scc/README.md`
