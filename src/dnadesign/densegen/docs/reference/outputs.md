## DenseGen Output Formats

DenseGen can emit USR datasets and/or Parquet datasets. Both formats share the same canonical ID
scheme and metadata semantics. Parquet is the canonical non-USR output format (columnar,
appendable, analytics-ready).

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

Parquet output is written as a single file (`outputs/dense_arrays.parquet`) under the run
root. Each row contains required fields (`id`, `sequence`, `bio_type`, `alphabet`, `source`)
plus namespaced `densegen__*` metadata columns.

Behavior:
- If `deduplicate: true`, existing IDs in the dataset are loaded and skipped.
- Parquet requires `pyarrow`; if unavailable, DenseGen fails fast.
- The output `path` must be a `.parquet` file (single-file output).
- List/dict metadata values are stored as native list/struct columns (no JSON encoding).
- If an existing dataset has a mismatched schema (for example, legacy JSON metadata), DenseGen
  fails fast and requires a fresh output path.
- DenseGen maintains a local ID index (`_densegen_ids.sqlite`) to speed deduplication and
  alignment checks.

---

### USR

USR output uses `Dataset.attach` with a fixed namespace (`densegen`). USR integration is
optional; if you do not include `usr` in `output.targets`, DenseGen should not import USR code.
USR output requires an explicit `output.usr.root`. List/dict metadata values are serialized to
JSON for attaches. USR output skips any IDs that already exist in `records.parquet` (resume-safe).

When multiple outputs are configured, all sinks must be in sync before a run. If one output
already exists and the other does not (or IDs differ), DenseGen fails fast.

---

### Metadata (common)

Keys are namespaced as `densegen__<key>`. Categories include:

- **Core + policy**: schema/run identifiers, solver/policy settings, compression ratio.
- **Inputs**: input type, input name, file/dataset source, PWM sampling metadata.
- **Constraints + postprocess**: fixed elements, promoter constraint tags, gap-fill policy.
- **Library + sampling**: library size, unique TF/TFBS counts, sampling caps and relaxations.
- **Placement stats**: used TFBS details, coverage of required regulators, per-TF counts.

Exact fields may expand over time. For the canonical list and types, see
`src/dnadesign/densegen/src/core/metadata_schema.py`.

---

### Run-level manifests

DenseGen writes run-level JSON files under `outputs/meta/`:

- `outputs/meta/run_state.json` — checkpointed progress for resumable runs (updated during execution).
- `outputs/meta/run_manifest.json` — summary counts per input/plan plus solver settings (written on completion). Includes a `leaderboard_latest` snapshot (top TF/TFBS usage, failure hotspots, diversity coverage).
- `outputs/meta/inputs_manifest.json` — resolved input paths and PWM sampling settings used for the run.

These are produced alongside Parquet/USR outputs and provide a compact audit trail.

---

### Library provenance (attempts log)

DenseGen now records solver library provenance exclusively in `outputs/attempts.parquet`.
Each attempt row stores the full library offered to the solver (`library_tfbs`, `library_tfs`,
`library_site_ids`, `library_sources`) along with the library hash/index and solver status.
Output records carry `densegen__sampling_library_hash` so you can join placements to attempts.

---

### Audit reports

The `dense report` command writes a compact audit summary under `outputs/`:

- `outputs/report.json`
- `outputs/report.md`

These summarize run scope and link to the canonical outputs (`dense_arrays.parquet` and
`attempts.parquet`).

---

### Source column

The `source` column is always present and encodes provenance as:

```
densegen:{input_name}:{plan_name}
```

Per-placement provenance (TFBS, offsets, orientations) is recorded in
`densegen__used_tfbs_detail` and the attempts log.

---

### Attempts log

DenseGen writes `outputs/attempts.parquet`, an append-only audit log of solver attempts
(success, duplicate, and constraint rejections). Each row includes the attempt status,
reason/detail JSON, solver metadata, and library hash/index. Each attempt also records the
exact library TF/TFBS/site_id lists offered to the solver (subset attribution). If no attempts
are logged, the file is absent. Attempts logs require `pyarrow`.

---

@e-south
