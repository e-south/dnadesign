## Outputs and metadata

DenseGen writes Parquet and/or USR outputs with a shared, deterministic ID scheme. Metadata is
namespaced and recorded consistently so outputs remain resume-safe and auditable.

### Contents
- [Output targets](#output-targets) - Parquet and USR sinks.
- [Run manifest](#run-manifest) - run-level summary JSON.
- [Effective config](#effective-config) - resolved config + derived caps/seeds.
- [Inputs manifest](#inputs-manifest) - resolved inputs and PWM sampling metadata.
- [Library manifest](#library-manifest) - libraries offered to the solver.
- [Rejection log](#rejection-log) - rejected solutions audit.
- [Source field](#source-field) - per-record provenance string.
- [Metadata scheme](#metadata-scheme) - namespacing and categories.
- [Parquet vs USR encoding](#parquet-vs-usr-encoding) - differences in storage.
- [Metadata registry](#metadata-registry) - canonical schema location.
- [Report assets](#report-assets) - plots emitted by `dense report`.

---

### Output targets

- **Parquet**: single-file `outputs/dense_arrays.parquet` plus audit Parquet tables
  (`outputs/attempts.parquet`, `outputs/solutions.parquet`, `outputs/composition.parquet`).
- **USR**: Dataset.attach with namespace `densegen`.

When multiple targets are configured, DenseGen asserts all targets are in sync before writing.

---

### Run manifest

Each run writes `outputs/meta/run_manifest.json` with per-input/plan counts (generated,
duplicates, failures, resamples, libraries built, stalls), derived seeds, solver settings,
schema version, and the dense-arrays version source. The manifest also tracks constraint-filter
failure reasons and duplicate-solution counts. A compact `leaderboard_latest` snapshot is recorded
per plan (top TF/TFBS usage, failure hotspots, and diversity coverage) for quick audits without
loading the full outputs.
Use the CLI to summarize a run:

```
uv run dense inspect run --run path/to/run
```

---

### Effective config

DenseGen writes `outputs/meta/effective_config.json`, which includes:
- fully-resolved config values (defaults expanded),
- derived seeds (`seed_stage_a`, `seed_stage_b`, `seed_solver`),
- resolved input paths, and
- computed sampling caps (requested candidates vs mining/time limits).

---

### Inputs manifest

When a run completes, DenseGen writes `outputs/meta/inputs_manifest.json`. This file captures
the resolved input paths (or dataset roots), PWM sampling settings, and the motif IDs actually
sampled so runs can be audited without re-opening the config or input files. PWM inputs include
per-motif site counts to make sampling behavior explicit.

---

### Stage‑A pools (TFBS pool artifacts)

DenseGen materializes Stage‑A pools under `outputs/pools/`:

- `outputs/pools/pool_manifest.json` — manifest of pool files by input.
- `outputs/pools/<input>__pool.parquet` — TFBS pools (or sequence pools).

TFBS pools include stable `motif_id` and `tfbs_id` hashes plus optional FIMO metadata
(`fimo_pvalue`, `fimo_bin_id`, etc.). Sequence pools include `tfbs_id` for joinability.

If `keep_all_candidates_debug: true`, DenseGen writes per-candidate debug artifacts under
`outputs/candidates/<run_id>/<input_name>/` (overwritten by `dense run` or `stage-a build-pool --overwrite`):
- `candidates__<label>.parquet` — candidate p‑values, bins, acceptance, and reject reasons.
- `<label>__fimo.tsv` — raw FIMO TSV (when enabled).
DenseGen also aggregates these into `outputs/candidates/<run_id>/candidates.parquet` and
`outputs/candidates/<run_id>/candidates_summary.parquet` with a manifest (`candidates_manifest.json`).
These are overwritten by `dense run` or `stage-a build-pool --overwrite`; copy the
`outputs/candidates/<run_id>` directory if you want to keep prior mining logs.

---

### Library artifacts (Stage‑B)

DenseGen writes Stage‑B libraries under `outputs/libraries/`:

- `library_builds.parquet` — one row per library build (index, hash, size, strategy).
- `library_members.parquet` — normalized membership table (one row per TFBS in each library).
- `library_manifest.json` — manifest + schema version.

These artifacts provide a stable join path from solver attempts to the exact library contents.

---

### Composition table

DenseGen writes `outputs/composition.parquet`, one row per TFBS placement in each accepted
sequence. Columns include `solution_id`, `attempt_id`, `input_name`, `plan_name`, `library_index`,
`tf`, `tfbs`, `motif_id`, `tfbs_id`, and placement offsets.

---

### Run state (checkpoint)

DenseGen writes `outputs/meta/run_state.json` during execution. This checkpoint captures
per-input/plan progress so long runs can resume safely after interruption.

---

### Events log

DenseGen writes `outputs/meta/events.jsonl` (JSON lines) with structured events:
`POOL_BUILT`, `LIBRARY_BUILT`, `STALL_DETECTED`, and `RESAMPLE_TRIGGERED`.
This is a lightweight, machine-readable trace of the run’s control flow.

---

### Report assets

`dense report` emits summary plots under `outputs/report_assets/` and links them in `report.html`.
These plots include Stage‑A p‑value/score histograms and Stage‑B utilization summaries. When
composition is available, the report also exports a full `composition.csv` under
`outputs/report_assets/`.

---

### Attempts log

DenseGen writes `outputs/attempts.parquet`, a consolidated log of solver attempts (success,
duplicate, and constraint rejections). Each row includes the attempt status, reason/detail JSON,
the sequence (if available), solver/provenance fields, and the exact library TF/TFBS/site_id lists
offered to the solver. Attempts include `attempt_id`, `attempt_index`, and (for successes)
`solution_id`. If no attempts occur, the file is absent. Attempts logs use Parquet and therefore
require `pyarrow`.

### Solutions log

DenseGen writes `outputs/solutions.parquet`, one row per accepted solution with `solution_id`,
`attempt_id`, and the library hash/index. Join keys: `solutions.solution_id` ↔ `dense_arrays.id`
and `solutions.attempt_id` ↔ `attempts.attempt_id`.

---

### Site failure summary

DenseGen does not write a separate site-failure table. Per-TFBS failure attribution can be
derived from `outputs/attempts.parquet` by grouping failures over the offered library lists.

---

### Source field

Every record includes a `source` string:

```
source = densegen:{input_name}:{plan_name}
```

This is always present and is separate from metadata.
Detailed placement provenance lives in `densegen__used_tfbs_detail` and the
run-scoped library manifests.
`densegen__used_tfbs_detail` includes `motif_id` and `tfbs_id` when available.

---

### Metadata scheme

All metadata keys are prefixed as `densegen__<key>`.

Typical categories:
- Provenance (`densegen__schema_version`, run identifiers, input info)
- Solver and policy (`densegen__solver_*`, `densegen__policy_*`)
- Library and sampling (`densegen__library_*`, `densegen__sampling_*`, `densegen__sampling_fraction*`)
- Constraints and postprocess (`densegen__fixed_elements`, `densegen__gap_fill_*`)
- Placement stats (`densegen__used_tfbs*`, `densegen__required_regulators*`)

`densegen__sampling_fraction` is defined as the fraction of **unique** TFBS strings in the
solver library divided by the realized input TFBS pool (`input_tfbs_count`).
`densegen__sampling_fraction_pairs` is the fraction of **unique TF:TFBS pairs** in the
solver library divided by `input_tf_tfbs_pair_count` (only meaningful for regulator-bearing inputs).

See `reference/outputs.md` for a fuller list and semantics.

---

### Parquet vs USR encoding

- Parquet stores list/dict metadata as native list/struct columns (no JSON encoding).
- USR stores list/dict metadata as JSON strings in attaches.

DenseGen fails fast if a Parquet dataset schema does not match the current registry.

---

### Metadata registry

DenseGen validates output metadata against a typed registry in
`src/dnadesign/densegen/src/core/metadata_schema.py` to keep fields stable and explicit as the
schema evolves.

---

@e-south
