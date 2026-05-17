## Study Records

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-14

Studies hold cross-tool context, motivation, notes, selected command surfaces,
and current handoff docs for long-running work. They are not generic workflow
docs and they are not the Ops API.

Study records are checked-in record artifacts for one live effort. They are not
runbooks or generated outputs. Treat them as the record plane: `ops progress`
may read them for observation, while `ops runbook` remains the control plane
for planning and execution.

Every checked-in study in this index carries an `ops.study.yaml` contract with
explicit `ops_surfaces.status_kind` and `ops_surfaces.preflight_kind` values.
That does not mean every request should invoke those commands: open-ended
design or product requests should still start from the route map, note, or
repo-local skill that owns the asked-for surface.

### Quick route

Use these surfaces by intent:

| Need | Surface | Why |
| --- | --- | --- |
| A request names a checked-in study | `docs/studies/<study-id>/routes.md` when present, otherwise the study directory README/status note | Start from the named study's living route map instead of assuming the repo-wide active study. |
| A request asks for status, history, blockers, or readiness | [Study Ops status surfaces](study-status-ops-surfaces.md) | Use Ops only when the task is explicitly observation or readiness. |
| A request asks how to add or refresh study records | [Study record authoring](study-record-authoring.md) | Keep authoring and promoter-template details out of this top-level router. |

For the checked-in Retron hairpin effort, start from
`docs/studies/retron_hairpin_design/routes.md` or the repo-local
`.agents/skills/retron-hairpin-study/SKILL.md` when the request is about MSD
labels, design references, visuals, or GenBank/FASTA outputs. Use pinned
`cruncher-study-status` and `cruncher-study-preflight` commands only for
explicit status or readiness questions.

### Authority chain

`docs/studies/index.yaml` selects the repo-wide active study. If the request
names a checked-in study that is not `active_study_id`, keep the selector
untouched and pin that study's directory or route map directly.

Each checked-in live study keeps these artifacts:

- `campaign.yaml`: explicit campaign/progress manifest when useful
- `datasets.yaml`: affiliated dataset registry and sync posture
- `status.md`: maintainer-facing current state and concise next actions
- `ops.study.yaml`: required OPS-facing lifecycle/track contract with explicit
  `ops_surfaces.status_kind` and `ops_surfaces.preflight_kind`
- `routes.md`: optional one-hop handoff map for current owner surfaces
- `pipeline.yaml`: optional machine-readable runtime context for exact command
  groups or automation bootstrap
- `audits/`: optional machine-readable sync/readiness evidence

Keep the code boundary clear: study status adapter implementation lives under
`src/dnadesign/studies/`, not under `src/dnadesign/ops/`. Ops reads checked-in
records and dispatches providers, but snapshot/preflight logic and
study-specific parsers stay with the status adapter or study package.

### Declared layout

```text
docs/studies/<study-id>/
  campaign.yaml
  datasets.yaml
  status.md
  ops.study.yaml
  routes.md      # optional, preferred once the study spans owner surfaces
  pipeline.yaml  # optional, for exact command groups/runtime context
  audits/
```

Study-specific implementation helpers, when needed, live under:

```text
src/dnadesign/studies/<study-id>/
```

Those helpers must stay narrow. If behavior becomes reusable outside one study,
promote it into the owning generic package or an explicitly shared contract
instead of growing a study-local tool.

### Related docs

- [Study status and preflight surfaces](study-status-ops-surfaces.md)
- [Study record authoring](study-record-authoring.md)
- [Study index](index.yaml)
- [Retron hairpin route map](retron_hairpin_design/routes.md)
- [Stress promoter route map](stress_ethanol_cipro_growth/routes.md)
- [Promoter study status contract](../../src/dnadesign/usr/docs/operations/promoter-study-status-contract.md)
- [Promoter study preflight contract](../../src/dnadesign/usr/docs/operations/promoter-study-preflight.md)
- [Documentation index](../README.md)
