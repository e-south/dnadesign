## Stress Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This lane stores machine-readable operating declarations for the
stress/ethanol/cipro growth study.

- `ops.study.yaml`: thin Ops contract router with part paths.
- `catalog/`: OPS status/preflight catalog pages and registry sidecars.
- `contract/`: lifecycle, phase, surface, status, and readiness declaration
  fragments loaded by `ops.study.yaml`. Execution command fragments are split
  by owner lane, with Notify subcommands under `contract/surfaces/execution/commands/notify/`.
  Large Infer readiness checks are split under
  `contract/readiness/checks/infer_batch_preparation/`.
- `runtime/command-groups/README.md`: progressive-disclosure entrypoint for
  runtime command groups.
- `runtime/command-groups/pipeline.yaml`: compatibility payload for command
  groups and downstream surface bindings.
- `runtime/command-groups/lanes/`: DenseGen, Infer, LatentDNA, Cluster, and
  OPAL navigation sidecars for agents that need one owner lane at a time.

Keep factual current state in `../record/status.md` and user-facing routing in
`../routes/README.md`.
