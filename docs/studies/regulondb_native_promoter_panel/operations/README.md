## RegulonDB Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This lane stores machine-readable operating declarations for the RegulonDB
native promoter panel study.

- `ops.study.yaml`: thin Ops contract router with part paths.
- `contract/`: lifecycle, phase, surface, status, and readiness declaration
  fragments loaded by `ops.study.yaml`.
- `runtime/command-groups/pipeline.yaml`: command groups and downstream surface
  bindings.

Keep factual current state in `../record/status.md` and user-facing routing in
`../routes/README.md`.
