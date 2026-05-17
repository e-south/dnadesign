## Study Status And Ops Surfaces

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14

Status, history, blockers, readiness, and Ops-backed automation enter through
the surfaces below. Navigation and design-product requests start from the named
study's `routes.md`.

### Surface model

- `docs/studies/index.yaml` is a repo-wide selector, not an ontology or router.
- `ops progress` is the observation surface for registered status/preflight
  routes.
- `ops.study.yaml` declares lifecycle and readiness shape through explicit
  `ops_surfaces.status_kind` and `ops_surfaces.preflight_kind` values.
- Not every request needs status/preflight. Open-ended design or product
  questions may still route to maps, notes, selected command groups, or
  repo-local skills.

### Common commands

- Active promoter-study snapshot:
  `uv run ops progress show usr.data-plane.promoter-study-status --json`
- Active promoter-study command preflight:
  `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --command-timeout-seconds 30 --json`
- Pin a named non-active study or run from outside the repo checkout:
  `--repo-root <repo-root> --study-dir docs/studies/<study-id>`

For the checked-in Retron hairpin study, use these only for explicit
status/readiness questions:

- `uv run ops progress show cruncher.data-plane.cruncher-study-status --study-dir docs/studies/retron_hairpin_design --json`
- `uv run ops progress show cruncher.data-plane.cruncher-study-preflight --study-dir docs/studies/retron_hairpin_design --scope next --json`

For Retron compiler/product requests, start from
`docs/studies/retron_hairpin_design/routes.md` or
`.agents/skills/retron-hairpin-study/SKILL.md` instead.

### Cold-thread procedure

1. Read `docs/studies/index.yaml`.
2. If the request names a checked-in study that is not `active_study_id`, pin
   that study's `routes.md` or status note first and treat the selector as
   discovery only.
3. Use `ops progress show ... --study-dir docs/studies/<study-id>` only when
   the question is explicitly about status, history, blockers, or readiness.
4. Return to `docs/studies/<study-id>/routes.md` after any state or blocker
   question is answered and the next owner surface is the real need.

### Failure rules

- If the registry and directory contents disagree, fail visibly and fix the
  registry before asking agents for live study status.
- If `routes.md` exists, treat it as the study-owned cross-tool handoff page
  rather than expanding the status note into a workflow encyclopedia.
- If `pipeline.yaml` exists, treat it as supplemental runtime context for exact
  command groups; do not reconstruct those paths from generic workspace docs.
