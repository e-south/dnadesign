## Study Status And Ops Surfaces

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Status, history, blockers, readiness, and Ops-backed automation enter through
the surfaces below. Navigation and design-product requests start from the named
study's `routes/README.md`.

### Surface model

- `docs/studies/index.yaml` is a repo-wide selector, not an ontology or router.
- `ops progress` is the observation surface for registered status/preflight
  routes.
- `operations/ops.study.yaml` declares lifecycle and readiness shape. It declares
  `ops_surfaces` only when the study owns concrete status/preflight providers.
- Not every request needs status/preflight. Open-ended design or product
  questions may still route to maps, notes, selected command groups, or
  repo-local skills.

### Common commands

- `stress_ethanol_cipro_growth` snapshot:
  `uv run ops progress show studies.stress-ethanol-cipro-growth.status --json`
- `stress_ethanol_cipro_growth` command preflight:
  `uv run ops progress show studies.stress-ethanol-cipro-growth.preflight --scope next --command-timeout-seconds 30 --json`
- Do not pin another study to this surface. A different study needs its own
  provider before it has an OPS status command.

For the checked-in Retron hairpin study, use these only for explicit
status/readiness questions:

- `uv run ops progress show studies.retron-hairpin-design.status --study-dir docs/studies/retron_hairpin_design --json`
- `uv run ops progress show studies.retron-hairpin-design.preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`

For Retron compiler/product requests, start from
`docs/studies/retron_hairpin_design/routes/README.md` or
`.agents/skills/retron-hairpin-study/SKILL.md` instead.

### Cold-thread procedure

1. Read `docs/studies/index.yaml`.
2. If the request names a checked-in study that is not `active_study_id`, pin
   that study's `routes/README.md` or `record/status.md` first and treat the
   selector as discovery only.
3. Use `ops progress show ...` only when the selected study owns the named
   status/preflight provider and the question is explicitly about status,
   history, blockers, or readiness.
4. Return to `docs/studies/<study-id>/routes/README.md` after any state or
   blocker question is answered and the next owner surface is the real need.

### Failure rules

- If the registry and directory contents disagree, fail visibly and fix the
  registry before asking agents for live study status.
- If `routes/README.md` exists, treat it as the study-owned cross-tool handoff
  page rather than expanding the status note into a workflow encyclopedia.
- If `operations/pipeline.yaml` exists, treat it as supplemental runtime
  context for exact command groups; do not reconstruct those paths from generic
  workspace docs.
