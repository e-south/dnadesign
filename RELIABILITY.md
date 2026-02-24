# RELIABILITY

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-19

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

## Operational signals
- USR event logs (`.events.log`) are the primary integration signal stream for watcher workflows.
- Cursor/spool state in notifier workflows must be explicit, restart-safe, and scoped to the intended workspace/run.
- Failures should include actionable context, not generic error wrappers.

## CI reliability lanes
- Core lane: lint/docs/format + standard-marker test selection + per-tool coverage gate; installs `ffmpeg` so baserender rendering tests run without environment skips.
- External integration lane: real FIMO/integration tests with explicit MEME/FIMO setup and verification; JUnit gate fails if all external integration tests are skipped and if any in-scope external integration tool executes zero non-skipped tests.
- CI gate lane: explicit merge gate that requires core lane success and requires external integration lane success whenever external integration scope is active.

## Operational runbook map
- SCC quickstart and batch guidance: `docs/bu-scc/quickstart.md`, `docs/bu-scc/batch-notify.md`
- SCC operational source of truth: `docs/bu-scc/sge-hpc-ops/SKILL.md`
- Notify operator runbook and event contracts: `docs/notify/README.md`, `docs/notify/usr-events.md`
- Maintainer CI/test details: `docs/dev/README.md`

## References
- Architecture map: `ARCHITECTURE.md`
- Security expectations: `SECURITY.md`
- Quality goals and measurements: `QUALITY_SCORE.md`
