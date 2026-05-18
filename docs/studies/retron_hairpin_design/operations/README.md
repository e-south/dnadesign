## Retron Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

This lane stores machine-readable operating declarations for the Retron hairpin
design study.

- `ops.study.yaml`: thin Ops contract router with part paths.
- `catalog/`: OPS status/preflight catalog pages and registry sidecars.
- `contract/`: lifecycle, track, surface, status, and readiness declaration
  fragments loaded by `ops.study.yaml`. Execution surfaces are split into
  `contract/surfaces/execution/{workspaces,commands}/`; readiness checks are
  split by Retron track under `contract/readiness/checks/`.
- `runtime/command-groups/README.md`: progressive-disclosure entrypoint for
  runtime command groups.
- `runtime/command-groups/pipeline.yaml`: compatibility payload for command
  groups and automation bootstrap metadata.
- `runtime/command-groups/lanes/`: compiler, materialize, Snapback, scar-nick,
  and YIU navigation sidecars for agents that need one owner lane at a time.

Keep factual current state in `../record/status.md` and user-facing routing in
`../routes/README.md`.
