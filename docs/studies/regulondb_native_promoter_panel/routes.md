## regulondb_native_promoter_panel Routes

**Last verified:** 2026-04-29

Use this page after the checked-in study status establishes the current phase.
This study remains inactive for production execution, but downstream
Construct/Infer/Notify/LatentDNA contracts are now checked in. The native USR
dataset and the Construct-derived TSS-upstream core60 dataset are materialized
locally and validated; generated data artifacts remain untracked.

- Status: `uv run ops progress show usr.data-plane.promoter-study-status --study-dir docs/studies/regulondb_native_promoter_panel --json`
- Preflight: `uv run ops progress show usr.data-plane.promoter-study-preflight --study-dir docs/studies/regulondb_native_promoter_panel --scope next --command-timeout-seconds 30 --json`

### Source Intake

- Type: `route`
- Plane: `data-plane`
- Surface role: `producer`
- Owner-boundary: `cruncher`
- Current state: `local_validated`
- Entry artifact: RegulonDB/EcoCyc live and release-pinned source payloads from declared source surfaces
- Exit artifact: deterministic Cruncher promoter export directory
- Primary code: `src/dnadesign/cruncher/src/ingest/promoters.py`
- Route note: Cruncher hides route quirks and carries source release, source table or API route, query checksum, raw checksum, missingness, source stratum, and conflict posture downstream. Local `dnadesign-data` sources are discovered through the public source-file API and recorded in `source_files.json` plus `source_file_inventory.json`; sequence-less curated rows are recorded in `skipped_source_rows.jsonl`; non-base strata are deferred rather than silently turned into duplicate sequence records.

#### Source Provenance

The source intake route must keep the following strata separate in the export
manifest and normalized source rows:

- `regulondb_13_promoter_set`: release-pinned curated completeness base from sibling `dnadesign-data`.
- `regulondb_14_5_live_graphql`: live GraphQL `getAllOperon` promoter route used as modern overlay and completeness check.
- `regulondb_13_sigmulon`: supplemental sigma promoter CSVs used for sigma-affiliation relations only.
- `regulondb_11_promoter_set`: historical curated release comparison.
- `regulondb_11_ht_tss`: RACE/454 high-throughput promoter/TSS evidence rows.
- `regulondb_11_prediction`: computational promoter predictions, kept as a typed prediction stratum and never promoted to curated truth by default.
- `ecocyc_28_promoters`: independent curated promoter cross-check with explicit window provenance.

Every source row must preserve the original release/source label and raw row
reference. The export must not merge local and live rows into one undifferentiated
source stratum.

### USR Import

- Type: `route`
- Plane: `data-plane`
- Surface role: `curated-dataset`
- Owner-boundary: `usr`
- Current state: `local_validated`
- Entry artifact: deterministic Cruncher promoter export directory
- Exit artifact: `usr_regulondb_native_promoters`
- Primary script: `src/dnadesign/usr/scripts/create_regulondb_native_promoters.py`
- First command: `uv run python src/dnadesign/usr/scripts/create_regulondb_native_promoters.py --export-dir <cruncher-promoter-export-dir>`
- Write command: `uv run python src/dnadesign/usr/scripts/create_regulondb_native_promoters.py --export-dir <cruncher-promoter-export-dir> --write`
- Route note: Dry-run is the default. Write mode refuses to overwrite an existing dataset. The generated dataset must have no duplicate canonical sequence rows, no duplicate source-alias rows, no non-ACGT sequence content, no orphan `usr_id` relation rows, and no sparse source-specific base columns.
- Readiness note: this route is complete in the current checkout when
  `src/dnadesign/usr/datasets/usr_regulondb_native_promoters/records.parquet`
  exists and `uv run usr validate usr_regulondb_native_promoters --strict`
  passes. Both checks passed locally on 2026-04-29.

The USR import creates dense `regulondb__*` overlay summaries on
`records.parquet`, source-record sequence views in
`_views/sequence_views.parquet`, mutable provenance/cohort semantics in
`_views/view_semantics.parquet`, and high-cardinality provenance facts in
relation sidecars inside the same dataset root.
Each retained native promoter sequence is exposed as
`product_kind=source_record`, not as an `analysis_window`; downstream core60 or
context views require an explicit derivation step.
Rows from curated source tables that lack usable DNA sequence are preserved in
`_relations/skipped_source_rows.parquet` rather than promoted to sequence base
rows.
Sequence-bearing source rows without sigma evidence are excluded from
`records.parquet` under the current strict source-panel policy and preserved in
`_relations/excluded_source_rows.parquet`.
Sigma summaries use canonical labels such as `sigma70`; raw source labels and
release-specific promoter identities remain in relation and source-row sidecars.
Fuzzy promoter-name matches are emitted as validation-report review signals and
must not be used as automatic sequence deduplication rules.

### Infer Native/Full 7B

- Type: `route`
- Plane: `control-plane`
- Surface role: `feature-extraction`
- Owner-boundary: `infer`
- Current state: `configured_preflight_ready`
- Entry artifact: `usr_regulondb_native_promoters`
- Exit artifact: `_derived/infer` sidecars under `usr_regulondb_native_promoters`
- Config: `src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml`
- Runbook: `src/dnadesign/ops/runbooks/presets/infer_regulondb_native_promoter_native_full_7b_batch_with_notify.yaml`
- Route note: This lane extracts native `source_record` views with
  `seq_mean` pooling and requests the intermediate block mean, output-layer
  mean, mean-per-token log likelihood, and total log likelihood sidecars.
  Local preflight validates the config, resolves the Notify event path, and
  reports all vectors/scalars missing as expected before Evo2 batch execution.

### Construct Native/Core60/Context

- Type: `route`
- Plane: `data-plane`
- Surface role: `derivation`
- Owner-boundary: `construct`
- Current state: `local_validated`
- Entry artifact: `usr_regulondb_native_promoters`
- Exit artifact: `usr_regulondb_native_promoter_core60`
- Workspace: `src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel`
- Route note: Native `source_record` rows remain source views. The checked-in
  core60 route emits a new `analysis_window` dataset by taking `[0,60)` from
  the native 81 bp source window, using the declared TSS offset `60`. This is
  not -10/-35 box centering. The 2026-04-29 materialization wrote 3,182
  sequence-view rows and 3,181 canonical 60 bp sequence rows; the row-count
  difference is expected USR sequence deduplication for duplicate derived
  windows.

### Infer Core60 TSS-Upstream 7B

- Type: `route`
- Plane: `control-plane`
- Surface role: `feature-extraction`
- Owner-boundary: `infer`
- Current state: `configured_preflight_ready`
- Entry artifact: `usr_regulondb_native_promoter_core60`
- Exit artifact: `_derived/infer` sidecars under `usr_regulondb_native_promoter_core60`
- Config: `src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml`
- Runbook: `src/dnadesign/ops/runbooks/presets/infer_regulondb_native_promoter_core60_tss_upstream_7b_batch_with_notify.yaml`
- Route note: This lane extracts derived `analysis_window` views with
  `core60_mean` pooling from the materialized core60 dataset. It
  requests the same intermediate block mean, output-layer mean, mean-per-token
  log likelihood, and total log likelihood sidecars as the native/full lane.

### Fill Remaining Infer

- Type: `route`
- Plane: `control-plane`
- Surface role: `batch-ergonomics`
- Owner-boundary: `ops`
- Current state: `plan_ready`
- Entry artifact: checked-in study `execution_surfaces` or explicit Infer runbook paths
- Exit artifact: one fill plan plus one workspace-scoped audit JSON per executed runnable lane
- Command: `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel`
- Submit command: `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel --submit`
- Route note: Ops discovers the study's Infer runbooks, runs the sequence-view
  completion inventory, skips complete lanes, blocks lanes with missing
  sequence products or stale sidecars, and plans only lanes with missing
  vectors/scalars. The primitive is study-record based and can also accept
  repeated `--runbook` paths, so it is not tied to RegulonDB or promoter
  semantics.

### LatentDNA Native Audit

- Type: `route`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `latentdna`
- Current state: `local_feature_review_ready`
- Entry artifact: native/full and later core60 7B vector and scalar feature surfaces
- Workspace: `src/dnadesign/latentdna/workspaces/regulondb_native_promoter_panel`
- Binding: `docs/studies/regulondb_native_promoter_panel/latentdna_binding.yaml`
- Route note: Native cohorts use `regulondb__*` fields. They must not derive
  DenseGen metadata or alias native sigma factors into `sig35_variant`.
  The native/full and core60 contracts both name intermediate embeddings,
  output-layer means, and log-likelihood scalar diagnostics from Infer sidecars.
  The current local snapshot is feature-backed and reports the primary
  decision deliverables as current; future missing sidecars must be expressed
  through explicit planned source roles, not hidden fallback behavior.
