# RELIABILITY

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-26

## At a glance
This document defines reliability intent for runtime behavior, CI behavior, and operator workflows.
It summarizes what must fail fast, what should be observable, and where recovery procedures live.

## Contents
- [Reliability model](#reliability-model)
- [Operational signals](#operational-signals)
- [CI reliability lanes](#ci-reliability-lanes)
- [Operational runbook map](#operational-runbook-map)
- [References](#references)

## Reliability model
- Missing hard prerequisites are fatal and explicit (for example absent required files/tools/config).
- Runtime and CI behavior should avoid hidden fallback paths.
- Long-running workflows should surface machine-readable state transitions and terminal outcomes.
- Repeated campaign orchestration state is workspace-scoped (`<workspace-root>/outputs/logs/ops/*`) to avoid root-level runbook/log fan-out.
- Observation-plane status discovery is metadata-first: checked-in tool-local
  `src/dnadesign/**/ops/status.registry.yaml` fragments and Ops-owned
  `src/dnadesign/ops/providers/*/status.registry.yaml` fragments must load
  without importing provider implementations, and provider code should only
  import when the selected surface executes.
- Snapshot and preflight surfaces stay distinct:
  - record-plane snapshots are cheap, repo-scoped, and should not probe local
    GPUs or scheduler state on the fast path
  - execution-readiness preflight is the authoritative deeper surface for
    host/workspace/cluster blockers before actual control-plane execution

## Operational signals
- USR event logs (`.events.log`) are the primary integration signal stream for watcher workflows.
- Cursor/spool state in notifier workflows must be explicit, restart-safe, and scoped to the intended workspace/run.
- Failures should include actionable context, not generic error wrappers.
- Checked-in study records under `docs/studies/<study-id>/` plus the active
  selector `docs/studies/index.yaml` are the authoritative record-plane signal
  for live-study posture.
- Status outputs should carry plane and summary-scope metadata so operators can
  distinguish repo snapshots from host-local readiness evidence.

## CI reliability lanes
- Core lane: lint/docs/format + standard-marker test selection + per-tool coverage gate; installs optional system packages only when in-scope tool tests require them.
- External integration lane: real FIMO/integration tests with explicit MEME/FIMO setup and verification; JUnit gate fails if all external integration tests are skipped and if any in-scope external integration tool executes zero non-skipped tests.
- CI gate lane: explicit merge gate that requires core lane success and requires external integration lane success whenever external integration scope is active.

## Operational runbook map
- SCC quickstart and batch guidance: `docs/bu-scc/quickstart.md`, `docs/bu-scc/batch-notify.md`
- SCC status-first and queue-fair operator guidance: `docs/bu-scc/quickstart.md`, `docs/bu-scc/batch-notify.md`, `docs/bu-scc/submission-reference.md`
- Repo BU SCC docs are the operational baseline; repo-local Codex skills are optional overlays, not required dependencies.
- Cross-tool orchestration and single-study accumulation contracts: `docs/operations/orchestration-runbooks.md`
- Notify operator runbook and event contracts: `docs/notify/README.md`, `docs/notify/usr-events.md`
- Maintainer CI/test details: `docs/dev/README.md`

## References
- Architecture map: `ARCHITECTURE.md`
- Security expectations: `SECURITY.md`
- Quality goals and measurements: `QUALITY_SCORE.md`
