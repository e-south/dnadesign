## Outputs and metadata

DenseGen writes Parquet outputs with a shared, deterministic ID scheme.
This guide is a short map; see `reference/outputs.md` for full schemas.

---

### Outputs layout (common)

```
outputs/
  tables/
    dense_arrays.parquet
    attempts.parquet
    solutions.parquet
    composition.parquet
    run_metrics.parquet
  pools/
  libraries/
  plots/
  report/
  meta/
  logs/
```

---

### What the artifacts mean (short)

- `tables/dense_arrays.parquet` - final sequences (canonical dataset).
- `tables/attempts.parquet` - solver attempt audit log.
- `tables/solutions.parquet` - accepted solutions (join to attempts).
- `tables/composition.parquet` - per-TFBS placements for accepted solutions.
- `tables/run_metrics.parquet` - run-level diagnostics.
- `pools/*` - Stage-A pools + manifests.
- `libraries/*` - Stage-B library artifacts.
- `meta/*` - run/inputs manifests, effective config, events.
- `plots/*` - plot images + plot manifest.

---

### Joining outputs (stable keys)

- `dense_arrays.parquet.id` -> `solutions.parquet.solution_id`
- `solutions.parquet.attempt_id` -> `attempts.parquet.attempt_id`
- `solutions.parquet.solution_id` -> `composition.parquet.solution_id`
- `attempts.parquet.library_index` -> `libraries/library_builds.parquet`

Stage-A pool joins are keyed by `tfbs_id` / `motif_id` where present.

---

### Metadata scheme

DenseGen metadata keys are prefixed with `densegen__*` and validated against a schema in
`src/dnadesign/densegen/src/core/metadata_schema.py`.

---

@e-south
