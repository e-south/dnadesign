## Retron Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This lane stores machine-readable operating declarations for the Retron hairpin
design study.

- `ops.study.yaml`: thin Ops contract router with part paths.
- `catalog/`: OPS status/preflight catalog pages and registry sidecars.
- `contract/`: lifecycle, track, surface, status, and readiness declaration
  fragments loaded by `ops.study.yaml`.
- `runtime/command-groups/pipeline.yaml`: command groups and automation
  bootstrap metadata.

Keep factual current state in `../record/status.md` and user-facing routing in
`../routes/README.md`.
