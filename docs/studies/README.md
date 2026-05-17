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

Checked-in studies carry an `ops.study.yaml` contract for lifecycle and owned
execution surfaces. A study declares `ops_surfaces.status_kind` and
`ops_surfaces.preflight_kind` only after it owns a concrete provider; do not
borrow another study's status surface.

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
`retron-hairpin-design-status` and `retron-hairpin-design-preflight` commands only for
explicit status or readiness questions.

### Authority chain

`docs/studies/index.yaml` selects the repo-wide active study. If the request
names a checked-in study that is not `active_study_id`, keep the selector
untouched and pin that study's directory or route map directly.

Each checked-in live study keeps these artifacts:

- `campaign.yaml`: optional explicit campaign/progress manifest when useful
- `datasets.yaml`: affiliated dataset registry and sync posture
- `status.md`: maintainer-facing current state and concise next actions
- `ops.study.yaml`: required OPS-facing lifecycle/track contract; optional
  `ops_surfaces` only when the study owns concrete status/preflight providers
- `routes.md`: optional one-hop handoff map for current owner surfaces
- `pipeline.yaml`: optional machine-readable runtime context for exact command
  groups or automation bootstrap
- `audits/`: optional machine-readable sync/readiness evidence

Keep the code boundary clear: concrete study status and preflight
implementation lives under `src/dnadesign/studies/studies/<study-id>/`, not under
`src/dnadesign/ops/` and not in a generic cross-study status bucket. Ops reads
checked-in records and dispatches providers, but snapshot/preflight logic and
study-specific parsers stay with the owning study package.

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
src/dnadesign/studies/studies/<study-id>/
```

Those helpers must stay narrow. If behavior becomes reusable outside one study,
promote it into the owning generic package or an explicitly shared contract
instead of growing a study-local tool.

### Related docs

- [Study status and preflight surfaces](study-status-ops-surfaces.md)
- [Study record authoring](study-record-authoring.md)
- [Study index](index.yaml)
- [Retron hairpin route map](retron_hairpin_design/routes.md)
- [Stress ethanol/cipro route map](stress_ethanol_cipro_growth/routes.md)
- [Stress ethanol/cipro status contract](stress_ethanol_cipro_growth/status-contract.md)
- [Stress ethanol/cipro preflight contract](stress_ethanol_cipro_growth/preflight.md)
- [Documentation index](../README.md)
