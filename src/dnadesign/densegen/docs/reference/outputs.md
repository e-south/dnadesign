## DenseGen outputs reference

This page defines what DenseGen writes, where it writes it, and which event stream each
consumer should read.

DenseGen can emit Parquet outputs locally and/or write through USR.

For a compact run-and-artifact checklist, use **[DenseGen quick checklist](../concepts/quick-checklist.md)**.

### Contents

This section covers contents.
- [Deterministic IDs](#deterministic-ids) - deterministic sequence identifiers.
- [Parquet](#parquet) - dataset layout and deduplication.
- [USR](#usr) - attach semantics and optional dependency.
- [Shared metadata fields](#shared-metadata-fields) - namespaced record fields and retention labels.
- [Run manifests](#run-manifests) - run-level JSON summaries.
- [Library provenance artifacts](#library-provenance-artifacts) - solver-offered libraries and attempts.
- [Source column values](#source-column-values) - provenance string.
- [Rejected solutions](#rejected-solutions) - constraint rejections audit.

---

### Event boundary paths

This section records event artifact paths only; the boundary explanation lives in
**[observability and events](../concepts/observability_and_events.md)**.

- DenseGen runtime diagnostics: `outputs/meta/events.jsonl`
- USR mutation events: `<usr_root>/<dataset>/.events.log`

---

### Deterministic IDs

This section covers deterministic ids.

- Sequence IDs use USR's algorithm:
  - `normalize_sequence(sequence, bio_type, alphabet)`
  - `compute_id(bio_type, normalized_sequence)`
- Parquet and USR must return the same `id` for the same sequence.
- `bio_type` and `alphabet` are defined once in `output.schema` and shared by all sinks.

---

### Parquet

Parquet output is written as a single file (`outputs/tables/records.parquet`) under the run root. Each row contains required fields (`id`, `sequence`, `bio_type`, `alphabet`, `source`) plus namespaced `densegen__*` metadata columns.

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

### Shared metadata fields

Keys are namespaced as `densegen__<key>`.

Retention labels in the table:
- `record_keep`: keep at record level for downstream analysis/provenance.
- `record_conditional`: keep at record level when applicable.
- `artifact_candidate`: better suited for run/library artifacts to avoid per-record redundancy.

Curated record fields:

| Field | Retention | Meaning |
|---|---|---|
| `densegen__schema_version` | artifact_candidate | DenseGen schema version (for example, 2.9). |
| `densegen__created_at` | record_keep | UTC ISO8601 timestamp for record creation. |
| `densegen__run_id` | record_keep | Run identifier (`densegen.run.id`). |
| `densegen__run_config_path` | artifact_candidate | Run config path (relative to run root when possible). |
| `densegen__length` | record_keep | Actual output sequence length. |
| `densegen__random_seed` | artifact_candidate | Global RNG seed used for the run. |
| `densegen__policy_sampling` | artifact_candidate | Stage-B sampling policy label (pool strategy). |
| `densegen__solver_backend` | artifact_candidate | Solver backend name (null when approximate). |
| `densegen__solver_strands` | artifact_candidate | Solver strands mode (`single`/`double`). |
| `densegen__dense_arrays_version` | artifact_candidate | `dense-arrays` package version. |
| `densegen__plan` | record_keep | Plan item name. |
| `densegen__tf_list` | record_keep | All TFs present in the Stage-B sampled library. |
| `densegen__tfbs_parts` | record_keep | `TF:TFBS` strings used to build the Stage-B library. |
| `densegen__used_tfbs` | record_keep | `TF:TFBS` strings used in the final sequence. |
| `densegen__used_tfbs_detail` | record_keep | Per-placement detail: `tf`/`tfbs`/`motif_id`/`tfbs_id`/orientation/offset plus Stage-A lineage fields. |
| `densegen__used_tf_counts` | record_keep | Per-TF placement counts (`{tf, count}`). |
| `densegen__covers_all_tfs_in_solution` | record_keep | Whether min-count TF coverage was satisfied. |
| `densegen__required_regulators` | record_keep | Regulators required for this library. |
| `densegen__min_count_by_regulator` | record_keep | Per-regulator minimum counts (`{tf, min_count}`). |
| `densegen__input_name` | record_keep | Input source name. |
| `densegen__input_mode` | record_keep | Input mode (`binding_sites`/`sequence_library`/`pwm_sampled`). |
| `densegen__input_pwm_ids` | record_conditional | Stage-A PWM motif IDs used for sampling (`pwm_*` inputs). |
| `densegen__input_tf_tfbs_pair_count` | record_conditional | Unique `(TF, TFBS)` pair count in the input pool. |
| `densegen__sampling_fraction` | record_conditional | Stage-B unique TFBS / input TFBS fraction. |
| `densegen__sampling_fraction_pairs` | record_conditional | Stage-B unique pair / input pair fraction. |
| `densegen__fixed_elements` | record_keep | Fixed-element constraints (promoters + side biases). |
| `densegen__visual` | record_keep | ASCII visual layout of placements. |
| `densegen__compression_ratio` | record_keep | Solution compression ratio. |
| `densegen__library_size` | record_keep | Number of motifs in the Stage-B sampled library. |
| `densegen__library_unique_tf_count` | record_keep | Unique TF count in sampled library. |
| `densegen__library_unique_tfbs_count` | record_keep | Unique TFBS count in sampled library. |
| `densegen__promoter_constraint` | record_conditional | Primary promoter constraint name. |
| `densegen__sampling_pool_strategy` | artifact_candidate | Stage-B sampling pool strategy. |
| `densegen__sampling_library_size` | artifact_candidate | Configured Stage-B library size. |
| `densegen__sampling_library_strategy` | artifact_candidate | Stage-B library sampling strategy. |
| `densegen__sampling_iterative_max_libraries` | artifact_candidate | Stage-B max libraries for iterative subsampling. |
| `densegen__sampling_library_hash` | record_keep | Stage-B stable sampled-library hash. |
| `densegen__sampling_library_index` | record_keep | Stage-B 1-based sampled library index. |
| `densegen__pad_used` | record_keep | Whether pad bases were applied. |
| `densegen__pad_bases` | record_conditional | Number of bases padded. |
| `densegen__pad_end` | record_conditional | Pad end (`5prime`/`3prime`). |
| `densegen__gc_total` | record_keep | GC fraction of final sequence. |
| `densegen__gc_core` | record_keep | GC fraction of pre-pad core sequence. |
| `densegen__npz_ref` | record_conditional | Relative NPZ artifact reference when `output.usr.npz_fields` is enabled (USR sink only). |
| `densegen__npz_sha256` | record_conditional | NPZ payload SHA256 (USR sink only). |
| `densegen__npz_bytes` | record_conditional | NPZ payload size in bytes (USR sink only). |
| `densegen__npz_fields` | record_conditional | Fields offloaded to NPZ (USR sink only). |

For type definitions and validation logic, see `src/dnadesign/densegen/src/core/metadata_schema.py`.

---

### Run manifests

DenseGen writes run-level JSON files under `outputs/meta/`:

- `outputs/meta/run_state.json` — checkpointed progress for resumable runs (updated during execution).
- `outputs/meta/run_manifest.json` — summary counts and quotas per input/plan plus solver settings and derived seeds (written on completion). Includes `total_quota`, `quota_progress_pct`, and per-item `quota`, plus a `leaderboard_latest` snapshot (top TF/TFBS usage, failure hotspots, diversity coverage).
- `outputs/meta/inputs_manifest.json` — resolved input paths and Stage‑A PWM sampling settings used for the run.
- `outputs/meta/effective_config.json` — resolved config with derived seeds and Stage‑A sampling caps.

These are produced alongside Parquet/USR outputs and provide a compact audit trail.

---

### DenseGen events log

DenseGen writes `outputs/meta/events.jsonl` (JSON lines) with structured events for pool builds, library builds, sampling pressure, stalls, and resamples. This is a lightweight machine-readable trace of runtime control flow.

---

### Related event docs

This section points to stream-boundary docs and keeps this file focused on artifact contracts.

See also:
- DenseGen boundary and mistakes: **[observability and events](../concepts/observability_and_events.md)**
- USR event schema: `../../../usr/README.md#event-log-schema`
- Notify operators doc: `../../../../../docs/notify/usr-events.md`
- DenseGen watcher walkthrough: **[DenseGen to USR to Notify tutorial](../tutorials/demo_usr_notify.md)**

---

### Run diagnostics table

DenseGen writes `outputs/tables/run_metrics.parquet`, an aggregated diagnostics table built from pools, libraries, attempts, and composition. It powers the run-level plots and makes it easy to answer: which libraries were weak, which TFs were over/under-used, and whether top-scoring Stage-A sites were actually consumed.
Tier-based metrics (e.g., tier enrichment) are computed only for PWM-derived pools with tier labels. Background pools do not carry tiers and are excluded from tier aggregations.

---

### Library provenance artifacts

DenseGen records solver library provenance in two places:

- `outputs/libraries/library_builds.parquet` + `library_members.parquet` (primary library artifacts).
- `outputs/tables/attempts.parquet` (attempt-level audit log with offered library lists). Each attempt row stores the full library offered to the solver (`library_tfbs`, `library_tfs`, `library_site_ids`, `library_sources`) along with the library hash/index and solver status. Attempts include `attempt_id` and `solution_id` (when successful) for stable joins. Output records carry both `densegen__sampling_library_hash` and `densegen__sampling_library_index`, and composition/solutions tables keep full library join keys.

---

### Solutions log

DenseGen writes `outputs/tables/solutions.parquet` (append-only) with the solution id, attempt id, and library hash. Join keys:

- `solutions.solution_id` ↔ `records.id`
- `solutions.attempt_id` ↔ `attempts.attempt_id`
- `solutions.solution_id` ↔ `composition.solution_id`

---

### Notebook output

`dense notebook generate` writes a workspace-scoped marimo notebook:

- `<run_root>/outputs/notebooks/densegen_run_overview.py` (default path)

The notebook reads run artifacts (`outputs/meta/*`, `outputs/tables/*`, `outputs/plots/plot_manifest.json`) and reflects the current persisted run state each time you open or rerun notebook cells.

DenseGen -> BaseRender contract for the scaffolded notebook:

- Contract source: `dnadesign.densegen:densegen_notebook_render_contract`
- Records source path: resolved from output sink selection:
  - single sink in `output.targets` -> that sink (`parquet` or `usr`)
  - multiple sinks in `output.targets` -> `plots.source`
  - `parquet` source path: `densegen.output.parquet.path`
  - `usr` source path: `<densegen.output.usr.root>/<densegen.output.usr.dataset>/records.parquet`
- BaseRender API surface used by the notebook:
  - `dnadesign.baserender.load_records_from_parquet(...)`
  - `dnadesign.baserender.render_record_figure(...)`
- Rendering style preset: `presentation_default` (fixed in the DenseGen contract)
- Notebook record preview window limit: `500` rows
- Required parquet columns for the default contract:
  - `id`
  - `sequence`
  - `densegen__used_tfbs_detail`
- Notebook layout (run mode):
  - `Workspace context`: workspace/config/output provenance and records path.
  - `Run summary`: compact status (`run_id`, `quota status`, `plans at quota`, solver backend/strategy).
  - `Records preview`: plan-filtered table with explicit dataset export controls (`parquet|csv`, output path).
  - `BaseRender preview`: `Prev`/`Next` controls with centered index/id status (`i / N | id: ...`).
- `Selected plot`: scope/type/plot controls stacked above the rendered plot.
- `Plot gallery`: filtered inventory table for browsing and audit.
- `Plot export`: export `selected|filtered|all` plots to one format (`pdf|png|svg`) under a target directory.
- Plot scope labels:
  - `all scopes (all plots)` includes run-level, stage-a, and plan-scoped artifacts.
  - `run-level` refers to unscoped run diagnostics (for example `run_health`).
  - `stage-a` refers to Stage-A pool diagnostics (`stage_a_summary`).
  - plan-scoped entries map to concrete plan names from `plot_manifest.json`.

---

### Plot output

`dense plot` writes plot images under `outputs/plots/` (format controlled by `plots.format`). `outputs/plots/plot_manifest.json` records the plot inventory and structured placement metadata for notebooks.

Run diagnostics metrics are summarized in `outputs/tables/run_metrics.parquet` (aggregated from pools, libraries, attempts, and composition). Plots below are generated from run artifacts plus accepted-sequence records loaded from the configured plot source (`plots.source`: parquet or usr), not candidate/debug logs.

Core diagnostics plots:

- `stage_a_summary` — Stage-A pool diagnostics (interpretation in the sampling guide).
- `placement_map` — Stage-B fingerprint: per-position occupancy/event counts across the final
  accepted sequences from `plots.source`, with overlaid categories for regulators and fixed elements (e.g., promoter -35/-10),
  showing where motifs land along the sequence.
- `run_health` — adaptive run diagnostics dashboard:
  outcome timeline by plan (discrete attempts for small runs, binned fractions for large runs),
  acceptance/waste/duplicate rates, rejected/failed reason Pareto, quota-aware accepted progress by plan,
  compression-ratio distribution by plan, and regulator-by-length TFBS usage counts.
- `tfbs_usage` — TFBS usage diagnostics from accepted placements: specific TFBS rank-count curve
  plus cumulative share by regulator.

Packaged DenseGen workspaces default to all four core plot families (`stage_a_summary`, `placement_map`, `run_health`, `tfbs_usage`).

`stage_a_summary` consolidates PWM inputs into one image per plot type (one row per input),
with outputs under `outputs/plots/stage_a/`:
- `pool_tiers.pdf`
- `yield_bias.pdf`
- `diversity.pdf`
- `<input>__background_logo.pdf` (background pools only)
- `no_stage_a_panels.pdf` (explicit no-op artifact when no Stage-A diagnostics apply to available pools)

`placement_map` writes one image under:
`outputs/plots/stage_b/<plan>/` (or `outputs/plots/stage_b/<plan>/<input>/` when multiple non-redundant inputs map to the same plan)
- `occupancy.pdf`

`tfbs_usage` writes one image into the same plan directory:
`outputs/plots/stage_b/<plan>/tfbs_usage.pdf` (or under `<input>/` for multi-input plans)

Stage-B scoping options are strict and per-plot:
- `plots.options.placement_map.scope: auto|per_plan|per_group`
- `plots.options.tfbs_usage.scope: auto|per_plan|per_group`
- `max_plans` (used by `auto`)
- `drilldown_plans` (optional per-plan detail count when grouped)

With `scope: auto`, DenseGen emits per-plan outputs for small plan sets and switches to grouped outputs when plan count exceeds `max_plans`. When grouping applies, plan-pool internal inputs are normalized to grouped plan scopes, so matrix-expanded runs do not fan out into one Stage-B file per expanded variant unless you request per-plan drilldowns.

`run_health` writes:
`outputs/plots/run_health/outcomes_over_time.pdf`
`outputs/plots/run_health/run_health.pdf`
`outputs/plots/run_health/compression_ratio_distribution.pdf`
`outputs/plots/run_health/tfbs_length_by_regulator.pdf`
`outputs/plots/run_health/summary_table.pdf`
`outputs/plots/run_health/summary.csv`

`run_health` uses status taxonomy `ok|rejected|duplicate|failed` from
`outputs/tables/attempts.parquet` and plan quotas from
`outputs/meta/effective_config.json` (`generation.plan[].sequences`).
`summary.csv` is a compact numeric table with run-level totals and per-plan accepted/quota ratios.
For large expanded runs, run-health plan rows auto-collapse to base plan groups unless overridden with plot style options (`run_health_plan_scope: per_plan|auto|per_group`, `run_health_plan_max_labels`).

See `../concepts/sampling.md` for plot interpretation context.

`stage_a_summary` requires diversity metrics in `pool_manifest.json`. If your pool manifest predates
diversity metrics, rerun `dense stage-a build-pool --fresh` to regenerate it.

---

#### How `placement_map` shows fixed elements

`placement_map` visualizes 1-nt occupancy across binding-site types (regulators and fixed elements).

When your plan includes `fixed_elements.promoter_constraints`, `placement_map` renders the promoter
as fixed-element bands overlaid alongside regulator occupancy so you can see how fixed constraints
consume positional budget relative to sampled sites.
Fixed elements are shown in the legend as `<name> -35` and `<name> -10`.

---

### Optional stage helper outputs

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

### Source column values

The `source` column is always present and encodes provenance as:

```
densegen:{input_name}:{plan_name}
```

Per-placement provenance (TFBS, offsets, orientations) is recorded in `densegen__used_tfbs_detail` (including `motif_id`/`tfbs_id`), `outputs/tables/composition.parquet`, and the attempts log.

---

### Rejected solutions

DenseGen records non-success solver attempts in `outputs/tables/attempts.parquet` with:

- `status` (`rejected`, `failed`, `duplicate`, `ok`)
- `reason` (normalized reason-family label)
- `reason_detail_json` (structured context for constraint audits)

This is the primary source for rejection audits and post-run failure triage.

---

### Attempts log

DenseGen writes `outputs/tables/attempts.parquet`, an append-only audit log of solver attempts (success, duplicate, and constraint rejections). Each row includes the attempt status, reason/detail JSON, solver metadata, and library hash/index. Each attempt includes:

- `attempt_id` — stable join key across artifacts
- `solution_id` — present for successful attempts
- `attempt_index` — per-plan monotonic counter

Each attempt also records the exact library TF/TFBS/site_id lists offered to the solver (subset attribution). If no attempts are logged, the file is absent. Attempts logs require `pyarrow`.
