## YIU Artifacts

**Audience:** YIU workflow users and maintainers
**Applies to:** `uv run cruncher yiu trace|solve|render`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-29

YIU writes two bundle families:

- explicit bundles under `outputs/yiu/explicit/<workflow>/<trace_id>/`
- solve bundles under `outputs/yiu/solve/<workflow>/<solve_id>/`

Both bundle families use one bundle-root render truth file:

- `visual_inventory.json`

`visual_inventory.json` is the single source of truth for:

- `protocol_template`
- view contract paths
- render artifact paths
- renderer kind
- state ids
- topology kinds
- render request and completion truth
- `render_count`
- `render_status`
- `last_rendered_at`

`cruncher yiu render --run <bundle>` rereads this file, regenerates the listed PDFs through the public BaseRender API, and writes the updated render truth back into the same inventory.

### Explicit bundle layout

```text
outputs/yiu/explicit/<workflow>/<trace_id>/
  report.json
  status.json
  manifest.json
  state_trace.jsonl
  visual_inventory.json
  tables/
    state_sequences.csv
    state_owners.csv
    effect_tags.csv
    fragment_summary.csv
  contracts/
    visuals/
      *.json
  visuals/
    *.pdf
```

Optional debug-only render jobs are written under `contracts/render_jobs/*.job.yaml` only when `persist_render_jobs_debug: true`.
Published view contracts live under `contracts/visuals/`.

### Solve bundle layout

```text
outputs/yiu/solve/<workflow>/<solve_id>/
  solve_report.json
  solve_status.json
  solve_manifest.json
  solution/
    report.json
    status.json
    manifest.json
    state_trace.jsonl
    visual_inventory.json
    tables/
      state_sequences.csv
      state_owners.csv
      effect_tags.csv
      fragment_summary.csv
    contracts/
      visuals/
        *.json
    visuals/
      *.pdf
  alternatives/
    solution_0002/
  comparison/
    solutions.csv
  visual_inventory.json
```

`alternatives/` and `comparison/` are present only when `compare_solutions: true`.

The solve-root `visual_inventory.json` points at the selected solution's view contracts and render artifacts. The selected solution remains the default operator story.
Selected-solution PDFs live under `solution/visuals/`.

### Status semantics

Explicit status:

- `satisfied` means the explicit validator accepted the authored spec
- `unsatisfied` means the bundle was still materialized but at least one hard issue remained

Solve status:

- `solved` means at least one admissible solution was found and the selected solution bundle was materialized
- `unsatisfied` means the bounded search completed without an admissible solution
- `incomplete_search` means the configured search limits were hit before exhaustion, so no public success was reported

### Operator inspection

`cruncher yiu show` surfaces:

- protocol template
- schema version or solve status
- final-state or selected-solution summary
- exhaustive-search truth for solve runs
- hard-invariant summary for the selected final state
- render summary from `visual_inventory.json`
- key artifact paths

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Spec Reference](yiu_spec.md) for schema details.
