## Study Records

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Studies hold cross-tool context, motivation, notes, selected command surfaces,
and current handoff docs for long-running work. They are not generic workflow
docs and they are not the Ops API.

Study records are checked-in record artifacts for one live effort. They are not
runbooks or generated outputs. Treat them as the record plane: `ops progress`
may read them for observation, while `ops runbook` remains the control plane
for planning and execution.

Checked-in studies carry an `operations/ops.study.yaml` contract for lifecycle
and owned execution surfaces. A study declares `ops_surfaces.status_kind` and
`ops_surfaces.preflight_kind` only after it owns a concrete provider; do not
borrow another study's status surface.

### Quick route

Use these surfaces by intent:

| Need | Surface | Why |
| --- | --- | --- |
| A request names a checked-in study | `docs/studies/<study-id>/routes/README.md` when present, otherwise the study directory README/status note | Start from the named study's living route map instead of assuming the repo-wide active study. |
| A request asks for status, history, blockers, or readiness | [Study Ops status surfaces](reference/study-status-ops-surfaces.md) | Use Ops only when the task is explicitly observation or readiness. |
| A request asks how to add or refresh study records | [Study record authoring](reference/study-record-authoring.md) | Keep authoring and promoter-template details out of this top-level router. |

For the checked-in Retron hairpin effort, start from
`docs/studies/retron_hairpin_design/routes/README.md` or the repo-local
`.agents/skills/retron-hairpin-study/SKILL.md` when the request is about MSD
labels, design references, visuals, or GenBank/FASTA outputs. Use pinned
`retron-hairpin-design-status` and `retron-hairpin-design-preflight` commands only for
explicit status or readiness questions.

### Authority chain

`docs/studies/index.yaml` selects the repo-wide active study. If the request
names a checked-in study that is not `active_study_id`, keep the selector
untouched and pin that study's directory or route map directly.

Each checked-in live study keeps these artifacts:

- `README.md`: directory ontology and first-hop usage note
- `record/`: checked-in factual record plane
  - `record/campaign.yaml`: optional explicit campaign/progress manifest
  - `record/datasets.yaml`: affiliated dataset registry and sync posture
  - `record/status.md`: maintainer-facing current state and concise next actions
- `operations/`: operational declarations read by tooling
  - `operations/ops.study.yaml`: required OPS-facing lifecycle/track contract;
    optional `ops_surfaces` only when the study owns concrete status/preflight
    providers
  - `operations/pipeline.yaml`: optional machine-readable runtime context for
    exact command groups or automation bootstrap
- `routes/README.md`: optional one-hop handoff map for current owner surfaces
- `routes/`: optional focused route-detail pages when one owner surface would
  otherwise turn the router into a workflow encyclopedia
- `contracts/`: optional status/preflight contracts and their registry sidecars
- `bindings/`: optional cross-tool study bindings such as LatentDNA context
- `contexts/`: optional long-form study rationale or handoff notes that are not
  current task routers
- `compiler/`: optional study-owned compiler input/config records when a study
  has a narrow compiler surface
- `workbench/`: optional study-specific experimental workbench for hypotheses,
  ontology terms, design sets, and run provenance that should outlive transient
  tool outputs. For multi-record workbenches, prefer `ontology/`,
  `design_sets/`, and `provenance/` lanes instead of placing YAML records flat
  at the workbench root.
- `audits/`: optional machine-readable sync/readiness evidence

Keep the code boundary clear: concrete study status and preflight
implementation lives under `src/dnadesign/studies/studies/<study-id>/`, not under
`src/dnadesign/ops/` and not in a generic cross-study status bucket. Ops reads
checked-in records and dispatches providers, but snapshot/preflight logic and
study-specific parsers stay with the owning study package.

### Declared layout

```text
docs/studies/<study-id>/
  README.md      # directory ontology and first-hop usage
  record/
    campaign.yaml
    datasets.yaml
    status.md
  operations/
    ops.study.yaml
    pipeline.yaml
  routes/
    README.md    # optional, preferred once the study spans owner surfaces
    ...          # optional focused route details for bulky owner surfaces
  contracts/     # optional, status/preflight docs and registry sidecars
  bindings/      # optional, cross-tool study bindings
  contexts/      # optional, long-form rationale and handoff notes
  compiler/      # optional, study-owned compiler inputs/config
  workbench/     # optional, durable ontology, design-set, and provenance records
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

- [Study reference docs](reference/README.md)
- [Study status and preflight surfaces](reference/study-status-ops-surfaces.md)
- [Study record authoring](reference/study-record-authoring.md)
- [Study index](index.yaml)
- [Retron hairpin route map](retron_hairpin_design/routes/README.md)
- [Stress ethanol/cipro route map](stress_ethanol_cipro_growth/routes/README.md)
- [Stress ethanol/cipro status contract](stress_ethanol_cipro_growth/contracts/status.md)
- [Stress ethanol/cipro preflight contract](stress_ethanol_cipro_growth/contracts/preflight.md)
- [Documentation index](../README.md)
