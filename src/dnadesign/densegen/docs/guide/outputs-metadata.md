## Outputs and metadata

This guide is a quick map of what DenseGen writes and how to join outputs.

For full schema details, use [../reference/outputs.md](../reference/outputs.md).

---

### Common outputs layout

```text
outputs/
  tables/
    records.parquet
    attempts.parquet
    solutions.parquet
    composition.parquet
    run_metrics.parquet
  pools/
  libraries/
  plots/
  meta/
  logs/
```

Workspace-level notebook artifact:

```text
outputs/notebooks/
  densegen_run_overview.py
```

---

### Subprocess -> artifact map

Read outputs by subprocess boundary:

1. **Stage-A pool build**
   - `outputs/pools/pool_manifest.json`
   - `outputs/pools/<input>__pool.parquet`
   - intent: what candidates were retained per input and why

2. **Stage-B library build**
   - `outputs/libraries/library_manifest.json`
   - `outputs/libraries/library_builds.parquet`
   - `outputs/libraries/library_members.parquet`
   - intent: what each plan solved against (the sampled library)

3. **Solve + runtime loop**
   - `outputs/tables/attempts.parquet`
   - `outputs/tables/solutions.parquet`
   - `outputs/tables/composition.parquet`
   - `outputs/tables/records.parquet`
   - intent: what was attempted, what was accepted, and final sequence composition

4. **Run metadata + diagnostics**
   - `outputs/meta/run_manifest.json`
   - `outputs/meta/effective_config.json`
   - `outputs/meta/events.jsonl`
   - `outputs/tables/run_metrics.parquet`
   - `outputs/plots/*`
   - intent: provenance, reproducibility, and health diagnostics

---

### What each artifact means

- `tables/records.parquet`: final sequences (canonical run dataset)
- `tables/attempts.parquet`: solver attempts (ok/rejected/duplicate/failed)
- `tables/solutions.parquet`: accepted solutions with stable join keys
- `tables/composition.parquet`: per-placement TFBS rows for accepted solutions
- `tables/run_metrics.parquet`: aggregated run diagnostics
- `pools/*`: Stage-A pools and manifest
- `libraries/*`: Stage-B library artifacts and manifest
- `meta/*`: run manifests, effective config, runtime events
- `plots/*`: rendered plots and plot manifest

---

### Stable join keys

- `records.id` -> `solutions.solution_id`
- `solutions.attempt_id` -> `attempts.attempt_id`
- `solutions.solution_id` -> `composition.solution_id`
- `attempts.library_index` -> `libraries/library_builds.parquet`

Stage-A joins use `tfbs_id` / `motif_id` where present.

Interpretation rule:
- `records` is the final accepted surface.
- `solutions` and `composition` explain how each final row was assembled.
- `attempts` explains why non-accepted work was rejected or failed.

---

### Metadata naming

DenseGen metadata fields are namespaced as `densegen__*` and validated by
`src/dnadesign/densegen/src/core/metadata_schema.py`.

Design intent:
- record-level fields carry sequence-specific provenance
- run-wide fields should stay in manifests/tables unless needed for filtering

---

@e-south
