## RegulonDB Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This lane stores machine-readable operating declarations for the RegulonDB
native promoter panel study.

- `ops.study.yaml`: thin Ops contract router with part paths.
- `contract/`: lifecycle, phase, surface, status, and readiness declaration
  fragments loaded by `ops.study.yaml`.
- `runtime/command-groups/README.md`: progressive-disclosure entrypoint for
  runtime command groups.
- `runtime/command-groups/pipeline.yaml`: compatibility payload for command
  groups and downstream surface bindings.
- `runtime/command-groups/lanes/`: source intake, USR import, Construct,
  Infer, and LatentDNA navigation sidecars for agents that need one owner lane
  at a time.

Keep factual current state in `../record/status.md` and user-facing routing in
`../routes/README.md`.
