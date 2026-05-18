## Stress Study Contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This directory holds Ops-facing contracts for the stress/ethanol/cipro growth
study. The root study record stays the entrypoint; these files define
observation and readiness surfaces.

- [Status contract](status.md): read-only study snapshot.
- [Preflight contract](preflight.md): read-only command and path readiness.
- `*.registry.yaml`: runbook-catalog sidecars for the adjacent contract docs.
