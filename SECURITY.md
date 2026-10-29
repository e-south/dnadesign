# SECURITY

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28

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
- Prefer secret backends and file-backed secret references for runtime secrets.
- Runtime tooling must keep secret material out of logs, command history, and generated configs where possible.
- Batch workflows should prefer file-backed secret references (`*_FILE` / `file://`) so secret values are not embedded in scheduler submit metadata.
- OPS omits supplied environment values from rendered plans and execution audits.
  Command arguments and captured output remain visible; commands must not embed
  or print credentials. Audit files use owner-only permissions.

## Dependency and supply-chain controls
- Python dependencies are pinned via `uv.lock` and installed with `uv sync --locked`.
- Pixi-managed external tools are pinned with `pixi.lock` and installed with `pixi install --locked`.
- Direct runtime imports have explicit published dependency floors; repository-only
  constraints are not treated as package metadata.
- GitHub Actions and remote pre-commit hooks are pinned to immutable commits.
- Pull requests that change dependency manifests run GitHub dependency review.
- CI and local workflows must avoid unpinned installs for operational paths.
- Release qualification audits all locked dependency profiles, including optional
  GPU dependencies. A passing CPU workflow does not clear an advisory in a GPU
  profile. Record the affected artifact, supported call path and remediation or
  mitigation evidence before changing a blocking security disposition.
- A source archive's hash does not identify binaries downloaded by its build
  script. Retain the actual installed artifact and build settings when reviewing
  native dependencies.

## Data handling expectations
- Treat dataset and run artifacts as operational data, not source-of-truth code.
- Scientific measurements and study interpretations belong to their study owner.
  Review tracked source as well as distribution contents before publication.
  Privacy scanning detects known identifiers and document digests; it does not
  establish that novel scientific content is appropriate for release.
- Do not hand-edit generated outputs; regenerate from code/config.
- Validate external sync/remote operations explicitly; do not assume side effects succeeded.

## Incident/reporting workflow
- Report suspected vulnerabilities privately through GitHub security
  advisories rather than a public issue.
- If a secret is discovered in repo history or working tree, stop and notify maintainers immediately.
- Rotate compromised tokens/webhooks first, then remediate repository history and runtime configs.
- Record corrective actions in the security issue and update the relevant runbooks.

## Enforcement controls
- Pre-commit hooks enforce secret hygiene and key detection (`.pre-commit-config.yaml`):
  - `detect-private-key`
  - `detect-secrets` with `.secrets.baseline`
- CI enforces secrets hygiene as a blocking lane (`.github/workflows/ci.yaml`):
  - `dnadesign.devtools.security.secrets_baseline` verifies `.secrets.baseline` paths still exist in the repo tree.
  - `pre-commit run detect-secrets --all-files` scans the full tracked tree against baseline policy.
- Core CI lane runs pre-commit checks on PR diff or full tree (`.github/workflows/ci.yaml`).
- CI validates workflow definitions using `check-github-workflows` via pre-commit configuration.
- GitHub secret scanning, push protection, Dependabot, and CodeQL complement
  the checked-in controls and must remain enabled in repository settings.

## References
- Root agent map and safety rules: `AGENTS.md`
- Notify operator manual: `docs/notify/README.md`
- SeqLedger events/operator contracts: `docs/notify/seqledger-events.md`
- BU SCC install and ops docs: `docs/bu-scc/setup/install.md`, `docs/bu-scc/README.md`
