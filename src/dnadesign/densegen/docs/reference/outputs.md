## DenseGen Output Formats

DenseGen can emit USR datasets and/or Parquet datasets. Both formats share the same canonical ID scheme and metadata semantics. Parquet is the canonical non-USR output format (columnar, appendable, analytics-ready).

### Contents
- [Canonical IDs](#canonical-ids) - deterministic sequence identifiers.
- [Parquet](#parquet) - dataset layout and deduplication.
- [USR](#usr) - attach semantics and optional dependency.
- [Metadata (common)](#metadata-common) - namespacing and core categories.
- [Run-level manifests](#run-level-manifests) - run-level JSON summaries.
- [Library manifests](#library-manifests) - libraries offered to the solver.
- [Source column](#source-column) - provenance string.
- [Rejected solutions](#rejected-solutions) - constraint rejections audit.

---

### Canonical IDs

- Sequence IDs use USR's algorithm:
  - `normalize_sequence(sequence, bio_type, alphabet)`
  - `compute_id(bio_type, normalized_sequence)`
- Parquet and USR must return the same `id` for the same sequence.
- `bio_type` and `alphabet` are defined once in `output.schema` and shared by all sinks.

---

### Parquet

Parquet output is written as a single file (`outputs/tables/dense_arrays.parquet`) under the run root. Each row contains required fields (`id`, `sequence`, `bio_type`, `alphabet`, `source`) plus namespaced `densegen__*` metadata columns.

Behavior:
- If `deduplicate: true`, existing IDs in the dataset are loaded and skipped.
- Parquet requires `pyarrow`; if unavailable, DenseGen fails fast.
- The output `path` must be a `.parquet` file (single-file output).
- List/dict metadata values are stored as native list/struct columns (no JSON encoding).
- If an existing dataset has a mismatched schema, DenseGen fails fast and requires a fresh output
  path.
- DenseGen maintains a local ID index (`outputs/meta/_densegen_ids.sqlite`) to speed deduplication and alignment checks.

---

### USR

USR output uses `Dataset.attach` with a fixed namespace (`densegen`). USR integration is
optional; if you do not include `usr` in `output.targets`, DenseGen should not import USR code. USR output requires an explicit `output.usr.root`. List/dict metadata values are serialized to JSON for attaches. USR output skips any IDs that already exist in `records.parquet` (resume-safe).

When multiple outputs are configured, all sinks must be in sync before a run. If one output already exists and the other does not (or IDs differ), DenseGen fails fast.

---

### Metadata (common)

Keys are namespaced as `densegen__<key>`. Categories include:

- **Core + policy**: schema/run identifiers, solver/policy settings, compression ratio.
- **Inputs**: input type, input name, file/dataset source, Stage‑A PWM sampling metadata.
- **Constraints + postprocess**: fixed elements, promoter constraint tags, pad policy.
- **Library + Stage‑B sampling**: library size, unique TF/TFBS counts, sampling caps and relaxations.
- **Placement stats**: used TFBS details, coverage of required regulators, per-TF counts.

Exact fields may expand over time. For the canonical list and types, see
`src/dnadesign/densegen/src/core/metadata_schema.py`.

---

### Run-level manifests

DenseGen writes run-level JSON files under `outputs/meta/`:

- `outputs/meta/run_state.json` — checkpointed progress for resumable runs (updated during execution).
- `outputs/meta/run_manifest.json` — summary counts per input/plan plus solver settings and derived seeds (written on completion). Includes a `leaderboard_latest` snapshot (top TF/TFBS usage, failure hotspots, diversity coverage).
- `outputs/meta/inputs_manifest.json` — resolved input paths and Stage‑A PWM sampling settings used for the run.
- `outputs/meta/effective_config.json` — resolved config with derived seeds and Stage‑A sampling caps.

These are produced alongside Parquet/USR outputs and provide a compact audit trail.

---

### Events log

DenseGen writes `outputs/meta/events.jsonl` (JSON lines) with structured events for pool builds, library builds, stalls, and resamples. This is a lightweight machine-readable trace of runtime control flow.

---

### Library provenance (library artifacts + attempts)

DenseGen records solver library provenance in two places:

- `outputs/libraries/library_builds.parquet` + `library_members.parquet` (canonical library artifacts).
- `outputs/tables/attempts.parquet` (attempt-level audit log with offered library lists). Each attempt row stores the full library offered to the solver (`library_tfbs`, `library_tfs`,
`library_site_ids`, `library_sources`) along with the library hash/index and solver status. Attempts include `attempt_id` and `solution_id` (when successful) for stable joins. Output records carry `densegen__sampling_library_hash` (Stage‑B) so you can join placements to libraries.

---

### Solutions log

DenseGen writes `outputs/tables/solutions.parquet` (append-only) with the canonical solution id, attempt id, and library hash. Join keys:

- `solutions.solution_id` ↔ `dense_arrays.id`
- `solutions.attempt_id` ↔ `attempts.attempt_id`
- `solutions.solution_id` ↔ `composition.solution_id`

---

### Audit reports

The `dense report` command writes a compact audit summary under `outputs/report/`:

- `outputs/report/report.json`
- `outputs/report/report.md`
- `outputs/report/report.html` (basic HTML wrapper for quick sharing)

These summarize run scope and link to the canonical outputs (`outputs/tables/dense_arrays.parquet` and
`outputs/tables/attempts.parquet`). Reports do not generate plots; run `dense plot` to populate
`outputs/plots/`, and use `dense report --plots include` to link the existing plot manifest.
Use `dense report --format json|md|html|all` to control which files are emitted.

---

### Plots

`dense plot` writes plot images under `outputs/plots/` (format controlled by `plots.format`).
`outputs/plots/plot_manifest.json` records the plot inventory for reports.

---

### Stage helper outputs (optional)

DenseGen can materialize Stage‑A/Stage‑B artifacts without running the solver:

- `dense stage-a build-pool` writes:
  - `outputs/pools/pool_manifest.json`
  - `outputs/pools/<input>__pool.parquet`
  - `outputs/pools/candidates/candidates__<label>.parquet` (when `keep_all_candidates_debug: true`)
  - `outputs/pools/candidates/candidates.parquet` + `candidates_summary.parquet` + `candidates_manifest.json`
    (overwritten by `dense run` or `stage-a build-pool --fresh`)
- `dense stage-b build-libraries` writes:
  - `outputs/libraries/library_builds.parquet`
  - `outputs/libraries/library_members.parquet`
  - `outputs/libraries/library_manifest.json`

`dense stage-a build-pool` appends new unique TFBS to existing pools by default; pass `--fresh`
to rebuild pools from scratch.

Stage‑B expects Stage‑A pools (default `outputs/pools`). These are optional inspection artifacts and are not required for a normal `dense run`.

---

### Source column

The `source` column is always present and encodes provenance as:

```
densegen:{input_name}:{plan_name}
```

Per-placement provenance (TFBS, offsets, orientations) is recorded in `densegen__used_tfbs_detail` (including `motif_id`/`tfbs_id`), `outputs/tables/composition.parquet`, and the attempts log.

---

### Attempts log

DenseGen writes `outputs/tables/attempts.parquet`, an append-only audit log of solver attempts (success, duplicate, and constraint rejections). Each row includes the attempt status, reason/detail JSON, solver metadata, and library hash/index. Each attempt includes:

- `attempt_id` — stable join key across artifacts
- `solution_id` — present for successful attempts
- `attempt_index` — per-plan monotonic counter

Each attempt also records the exact library TF/TFBS/site_id lists offered to the solver (subset attribution). If no attempts are logged, the file is absent. Attempts logs require `pyarrow`.

---

@e-south
