## YIU Artifacts

**Audience:** YIU workflow users and maintainers
**Applies to:** `uv run cruncher yiu design|trace|solve`
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27

YIU now writes two bundle families:

- explicit bundles under `outputs/yiu/explicit/<spec.name>/<design_id>/`
- solve bundles under `outputs/yiu/solve/<solve_name>/<solve_id>/`

### Explicit bundle

Explicit runs write:

- `yiu_report.json`
- `yiu_status.json`
- `yiu_manifest.json`
- `yiu_trace.jsonl`
- `yiu_trace_manifest.json`
- `yiu_parts.csv`
- `yiu_annotations.csv`
- `yiu_fragments.csv`
- `published/views/`
- `published/visual_manifest.json`
- `published/baserender_jobs/` when job emission is enabled
- `published/renders/` as the target directory for emitted BaseRender jobs

Explicit `yiu_status.json`, `yiu_manifest.json`, and `yiu_report.json` preserve canonical `protocol_template` and any deprecated alias metadata when the run uses a `schema_version: 2` template-driven spec.

`published/visual_manifest.json` is the operator-facing visual inventory. It unifies:

- neutral state views
- render-oriented YIU contracts
- BaseRender job files
- rendered output locations
- bundle-local view/job/render counts

`yiu_manifest.json` now also includes:

- `machine_artifacts`
- `published_artifacts`
- `artifacts` entries that enumerate the real publication layer on disk

The split-template explicit lane emits render-oriented contracts such as:

- `yiu_linear_state_v1`
- `yiu_hairpin_topology_v1`
- `yiu_topology_cartoon_v1`

### Solve bundle

Solve runs write:

```text
outputs/yiu/solve/<solve_name>/<solve_id>/
  yiu_solve_report.json
  yiu_solve_status.json
  yiu_solve_manifest.json
  hits.csv
  accepted_hits.jsonl
  published/
    views/
    baserender_jobs/
    renders/
    visual_manifest.json
  hits/
    hit_0001/
      ... standard explicit YIU bundle ...
```

`accepted_hits.jsonl` is the stable machine-readable solve hit stream. Each materialized hit includes:

- `rank`
- `hit_id`
- `score`
- `source_sequence`
- `materialized_run_dir`
- `final_state_id`
- solve-level published view/job paths when available

`yiu_solve_manifest.json` now also includes:

- `published_artifacts`
- `hit_bundle_root`
- `top_hit_ids`
- `materialized_hit_bundle_roots`
- `copied_top_hit_artifacts`
- `hits_csv`
- `accepted_hits_stream`

### Status semantics

Explicit status:

- `completed` means the explicit validator satisfied the requested spec
- `unsatisfied` means the bundle was still materialized, but at least one hard issue remains

Solve status:

- `solved` means at least one hit passed the explicit validator as the final oracle
- `no_hits` means search completed without an admissible hit
- `invalid_spec` is reserved for solve-spec or base-spec preflight failures

### Operator inspection

`cruncher yiu show` reads both bundle families and surfaces:

- run root
- bundle kind
- status
- run id or solve id
- step/state/issue counts for explicit bundles
- accepted/materialized hit counts for solve bundles
- `published/views/`
- `published/visual_manifest.json`
- `published/baserender_jobs/`
- `published/renders/`
- solve-level `accepted_hits.jsonl` and the first hit path when the run is a solve bundle
- top-hit explicit bundle roots when the run is a solve bundle

`cruncher yiu show --json` emits the normalized artifact inventory that the CLI text view summarizes.

Use [YIU Workflow](../guides/yiu_workflow.md) for execution guidance and [YIU Spec Reference](yiu_spec.md) for schema details.
