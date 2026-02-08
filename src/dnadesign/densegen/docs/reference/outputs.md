## DenseGen Output Formats

DenseGen can emit Parquet datasets (stored locally or in the sibling USR package).

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

### Integration boundary

DenseGen runtime diagnostics and USR mutation events are separate streams:

- DenseGen runtime diagnostics: `outputs/meta/events.jsonl`
- USR mutation events: `<usr_root>/<dataset>/.events.log`

Notify reads USR `.events.log`, not DenseGen `outputs/meta/events.jsonl`.

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

USR output uses the USR Dataset overlay-part writer with a fixed namespace (`densegen`). USR integration is
optional; if you do not include `usr` in `output.targets`, DenseGen should not import USR code. USR output requires
an explicit `output.usr.root` and a registry entry for the `densegen` namespace (repo-wide `usr/datasets/registry.yaml`).

Behavior:
- Base sequences are imported once; derived metadata is written as overlay parts under `_derived/densegen/part-*.parquet`.
- Metadata values are stored as typed Arrow list/struct columns (no JSON encoding).
- Existing IDs in `records.parquet` are skipped (resume-safe).
- Overlay parts are append-only. Compact parts with `usr maintenance overlay-compact <dataset> --namespace densegen`.
 - If `output.usr.npz_fields` is set, those metadata fields are offloaded into NPZ artifacts under
   `<dataset>/_artifacts/densegen_npz/<id>.npz`. Overlay columns `densegen__npz_ref`,
   `densegen__npz_sha256`, `densegen__npz_bytes`, and `densegen__npz_fields` are populated and the
   offloaded fields are stored as null inline.

When multiple outputs are configured, all sinks must be in sync before a run. If one output already exists and the other
does not (or IDs differ), DenseGen fails fast.

---

### Metadata (common)

Keys are namespaced as `densegen__<key>`. Categories include:

- **Core + policy**: schema/run identifiers, solver/policy settings, compression ratio.
- **Inputs**: input type, input name, file/dataset source, Stage‑A PWM sampling metadata.
- **Constraints + postprocess**: fixed elements, promoter constraint tags, pad policy.
- **Library + Stage‑B sampling**: library size, unique TF/TFBS counts, sampling caps and relaxations.
- **Placement stats**: used TFBS details, coverage of required regulators, per-TF counts.

Exact fields may expand over time. For the canonical list and types, see `src/dnadesign/densegen/src/core/metadata_schema.py`.

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

DenseGen writes `outputs/meta/events.jsonl` (JSON lines) with structured events for pool builds, library builds, sampling pressure, stalls, and resamples. This is a lightweight machine-readable trace of runtime control flow.

---

### Event streams and consumers (DenseGen vs USR)

DenseGen participates in two distinct event streams when you enable the USR sink:

| Stream | Path | Producer | Primary purpose | Typical consumer |
|---|---|---|---|---|
| DenseGen runtime events | `outputs/meta/events.jsonl` | DenseGen | Run diagnostics (resamples, stalls, library rebuilds) | `dense inspect run --events`, plots, reports |
| USR mutation events | `<usr_root>/<dataset>/.events.log` | USR | Audit plus integration boundary | `notify usr-events watch`, `usr events tail` |

Important: Notify consumes USR `.events.log`, not DenseGen `outputs/meta/events.jsonl`.
See also:
- USR event schema: `../../../usr/README.md#event-log-schema`
- Notify operators doc: `../../../../../docs/notify/usr_events.md`

---

### Run diagnostics table

DenseGen writes `outputs/tables/run_metrics.parquet`, an aggregated diagnostics table built from pools, libraries, attempts, and composition. It powers the run-level plots and makes it easy to answer: which libraries were weak, which TFs were over/under-used, and whether top-scoring Stage-A sites were actually consumed.
Tier-based metrics (e.g., tier enrichment) are computed only for PWM-derived pools with tier labels. Background pools do not carry tiers and are excluded from tier aggregations.

---

### Library provenance (library artifacts + attempts)

DenseGen records solver library provenance in two places:

- `outputs/libraries/library_builds.parquet` + `library_members.parquet` (canonical library artifacts).
- `outputs/tables/attempts.parquet` (attempt-level audit log with offered library lists). Each attempt row stores the full library offered to the solver (`library_tfbs`, `library_tfs`, `library_site_ids`, `library_sources`) along with the library hash/index and solver status. Attempts include `attempt_id` and `solution_id` (when successful) for stable joins. Output records carry `densegen__sampling_library_hash` (Stage‑B) so you can join placements to libraries.

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

These summarize run scope and link to the canonical outputs (`outputs/tables/dense_arrays.parquet` and `outputs/tables/attempts.parquet`). Reports do not generate plots; run `dense plot` to populate `outputs/plots/`, and use `dense report --plots include` to link the existing plot manifest. Use `dense report --format json|md|html|all` to control which files are emitted.

---

### Plots

`dense plot` writes plot images under `outputs/plots/` (format controlled by `plots.format`). `outputs/plots/plot_manifest.json` records the plot inventory for reports.

Run diagnostics metrics are summarized in `outputs/tables/run_metrics.parquet` (aggregated from pools, libraries, attempts, and composition). Plots below are generated directly from canonical artifacts (Parquet + manifests), not candidate/debug logs.

Core diagnostics plots (canonical set):

- `stage_a_summary` — Stage-A pool diagnostics (interpretation in the sampling guide).
- `placement_map` — Stage-B fingerprint: per-position occupancy/event counts across the final
  dense arrays, with overlaid categories for regulators and fixed elements (e.g., promoter -35/-10),
  plus a TFBS allocation view (rank–frequency + cumulative share).

Optional / advanced:
- `run_health` — attempts outcomes + failure composition. (Not a default plot; prefer `dense inspect run`.)
- `tfbs_usage` — legacy TFBS allocation view (superseded by placement_map’s allocation plot).

`stage_a_summary` consolidates PWM inputs into one image per plot type (one row per input),
e.g. `stage_a_summary__pool_tiers.pdf`, `stage_a_summary__yield_bias.pdf`,
and `stage_a_summary__diversity.pdf`.
Background pools emit a separate logo image per background input
(`stage_a_summary__<input>__background_logo.pdf`).

`placement_map` writes two images per input/plan:
`placement_map__<input>__<plan>__occupancy.pdf` and
`placement_map__<input>__<plan>__tfbs_allocation.pdf`.

See `../guide/sampling.md` for plot interpretation context.

`stage_a_summary` requires diversity metrics in `pool_manifest.json`. If your pool manifest predates
diversity metrics, rerun `dense stage-a build-pool --fresh` to regenerate it.

---

#### `placement_map` overlays fixed elements

`placement_map` visualizes 1-nt occupancy across binding-site types (regulators and fixed elements).

When your plan includes `fixed_elements.promoter_constraints`, `placement_map` renders the promoter
as fixed-element bands overlaid alongside regulator occupancy so you can see how fixed constraints
consume positional budget relative to sampled sites.
Fixed elements are shown in the legend as `<name> -35` and `<name> -10`.

---

### Stage helper outputs (optional)

DenseGen can materialize Stage‑A/Stage‑B artifacts without running the solver:

- `dense stage-a build-pool` writes:
  - `outputs/pools/pool_manifest.json`
  - `outputs/pools/<input>__pool.parquet`
  - `outputs/pools/candidates/<input_name>/candidates__<label>.parquet` (when `keep_all_candidates_debug: true`)
  - `outputs/pools/candidates/<input_name>/<label>__fimo.tsv` (when `keep_all_candidates_debug: true`)
  - `outputs/pools/candidates/candidates.parquet` + `candidates_summary.parquet` + `candidates_manifest.json`
    (overwritten by `stage-a build-pool --fresh` or `dense run --fresh`)
- `dense stage-b build-libraries` writes:
  - `outputs/libraries/library_builds.parquet`
  - `outputs/libraries/library_members.parquet`
  - `outputs/libraries/library_manifest.json`

`dense stage-a build-pool` appends new unique TFBS to existing pools by default; pass `--fresh`
to rebuild pools from scratch.
`pool_manifest.json` includes the input config hash plus file fingerprints; append requires they match.
For FIMO-backed PWM inputs, it records Stage-A sampling metadata, including:
- tier fractions + source + scheme label
- eligibility/retention rules, FIMO threshold, background source/bgfile
- consensus and max-score stats (`pwm_consensus`, `pwm_consensus_iupac`, `pwm_consensus_score`,
  `pwm_theoretical_max_score`, `max_observed_score`)
- selection pool diagnostics (`selection_pool_*`, `selection_score_norm_*`)
- trimming metadata (`motif_width`, `trimmed_width`, `trim_window_length`,
  `trim_window_strategy`, `trim_window_start`, `trim_window_score`, `trim_window_applied`)
- diversity summaries (k‑NN unweighted/weighted, pairwise weighted, overlap, score quantiles)
- mining and padding audits

See the sampling guide for interpretation; the manifest is the source of truth for field names.
Stage‑A pool rows include `best_hit_score`, `tier`, `rank_within_regulator` (1‑based rank among
eligible_unique TFBS per regulator), and `tfbs_core` for core‑level uniqueness checks.

Stage‑B expects Stage‑A pools (default `outputs/pools`). `dense run` reuses these pools by default
and rebuilds them if they are missing or stale; rebuild explicitly with `dense stage-a build-pool --fresh`
or reset everything with `dense run --fresh`.

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
