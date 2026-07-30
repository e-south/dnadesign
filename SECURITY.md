# SECURITY

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-30

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

## Dependency and supply-chain controls
- Python dependencies are pinned via `uv.lock` and installed with `uv sync --locked`.
- Pixi-managed external tools are pinned with `pixi.lock` and installed with `pixi install --locked`.
- Direct runtime imports have explicit published dependency floors; repository-only
  constraints are not treated as package metadata.
- GitHub Actions and remote pre-commit hooks are pinned to immutable commits.
- Pull requests that change dependency manifests run GitHub dependency review.
- CI and local workflows must avoid unpinned installs for operational paths.

### Bounded dependency exceptions

Three upstream advisories across two constrained dependencies cannot yet be
resolved without breaking an owning runtime constraint. They remain open,
visible, and time-bounded:

- `torch` is constrained to the Evo2-compatible 2.10 series.
  GHSA-53q9-r3pm-6pq6 affects restricted `torch.load` deserialization before
  2.6 and is fixed in the supported series. Every repository-owned
  `torch.load` call still sets `weights_only=True`; loaders reject unsupported
  globals instead of falling back to unrestricted pickle.
  PYSEC-2026-139 affects PT2 loading through 2.10.0. PT2 artifacts and
  `torch.export.load` are not supported serialization surfaces.
  GHSA-rrmf-rvhw-rf47 affects `torch.jit.script` and is fixed in 2.13.
  Repository-owned code does not call that function. Reassess when the
  Evo2/CUDA stack supports PyTorch 2.13, and no later than 2026-10-29.
- `pymdown-extensions` 10.21.3 is constrained by Marimo.
  GHSA-9xwg-3r6f-jcx2 affects `pymdownx.b64`. Repository-owned code does not
  enable that extension, and supported execution is native macOS or Linux.
  Upstream Marimo can enable it under Pyodide/WASM, which is therefore outside
  the supported deployment boundary while this exception is active. Reassess
  when Marimo permits version 11, and no later than 2026-08-29.

The unresolved exception statements narrow supported reachability; they do not
claim that affected installed versions are patched. Do not dismiss the
corresponding alerts.

## Data handling expectations
- Treat dataset and run artifacts as operational data, not source-of-truth code.
- Do not hand-edit generated outputs; regenerate from code/config.
- Validate external sync/remote operations explicitly; do not assume side effects succeeded.

## Incident/reporting workflow
- Report suspected vulnerabilities privately through GitHub security
  advisories rather than a public issue.
- If a secret is discovered in repo history or working tree, stop and notify maintainers immediately.
- Rotate compromised tokens/webhooks first, then remediate repository history and runtime configs.
- Capture corrective actions in maintainer notes (`docs/dev/journal.md`) and relevant runbooks.

## Enforcement controls
- Pre-commit hooks enforce secret hygiene and key detection (`.pre-commit-config.yaml`):
  - `detect-private-key`
  - `detect-secrets` with `.secrets.baseline`
- CI enforces secrets hygiene as a blocking lane (`.github/workflows/ci.yaml`):
  - `dnadesign.devtools.security.secrets_baseline` verifies `.secrets.baseline` paths still exist in the repo tree.
  - `pre-commit run detect-secrets --all-files` scans the full tracked tree against baseline policy.
- Core CI lane runs pre-commit checks on PR diff or full tree (`.github/workflows/ci.yaml`).
- CI validates workflow definitions using `check-github-workflows` via pre-commit configuration.
- CI checks that bounded exceptions remain inside the documented native
  deployment boundary and that their review dates have not expired.
- GitHub secret scanning, push protection, Dependabot, and CodeQL complement
  the checked-in controls and must remain enabled in repository settings.

## References
- Root agent map and safety rules: `AGENTS.md`
- Notify operator manual: `docs/notify/README.md`
- USR events/operator contracts: `docs/notify/usr-events.md`
- BU SCC install and ops docs: `docs/bu-scc/setup/install.md`, `docs/bu-scc/README.md`
