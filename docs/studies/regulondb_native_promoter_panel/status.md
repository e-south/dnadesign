## regulondb_native_promoter_panel

- Last verified: 2026-04-30
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Route map: `routes.md`
- Study execution map: `pipeline.yaml`
- USR root: `src/dnadesign/usr/datasets`
- Lifecycle posture: inactive source-intake lane; local native USR and TSS-upstream core60 datasets are materialized, and the standard local Evo2 7B Infer sidecars are complete for the native/full and derived core60 lanes

### Current Datasets

- Native promoter source: `usr_regulondb_native_promoters` (`local validated`, generated/untracked)
- Native/full 7B vector/scalar sidecars: `infer_regulondb_native_promoter_views_7b` (`local complete`)
- Core60 view: `usr_regulondb_native_promoter_core60` (`local validated`, generated/untracked)
- Core60 7B vector/scalar sidecars: `infer_regulondb_native_promoter_core60_views_7b` (`local complete`)
- Native/full plus in-context TSS-upstream core60 sidecars:
  `config.sequence_views.native_full_plus_tss_upstream_core60.evo2_7b.yaml`
  (`local complete`, additive dogfood lane, not part of the default fill quota)

### Current Phase

- Declared phase: `local_infer_complete_7b`
- Source export status: local Cruncher superset export validated; export artifacts are not checked in
- USR dataset status: `src/dnadesign/usr/datasets/usr_regulondb_native_promoters` is materialized locally and passes strict USR validation as of 2026-04-29
- Sequence-view status: write mode now emits one `source_record` sequence view per retained native promoter sequence, plus mutable view semantics for `source_family`, `selection_basis`, `view_collections`, and `role_tags`
- Preferred first infer family: `evo2_7b`
- Current batch ergonomics: `ops runbook fill-infer` inspects the checked-in
  Infer runbooks, skips complete sequence-view vector/scalar lanes, and now
  plans zero RegulonDB GPU submissions from this checkout.

### Placement Boundary

This study is a data intake and validation lane, not a new top-level dnadesign
tool. Source file discovery uses the sibling `dnadesign-data` public API.
Cruncher normalizes source rows into a fixed export. USR turns that fixed export
into one sequence-level dataset after validation passes. Generated exports, USR
parquet sidecars, snapshots, and downstream embeddings stay out of the
checked-in study record. The checked-in record describes how the dataset was
built and validated; the generated USR dataset itself is managed as a
local/shared data artifact, not as git-tracked source.

### Provenance Model

`usr_regulondb_native_promoters` is the single USR dataset for the strict native
promoter superset. It is not a live-only dataset. Its input is a fixed Cruncher
superset export that records source, release, table or API route, row reference,
raw checksum, parser version, and source stratum before USR import.

Source strata represented by the current validated export:

| Stratum | Role | Provenance rule |
| --- | --- | --- |
| RegulonDB 13 `PromoterSet.tsv` | Current curated release-pinned base source | Rows with strict DNA sequence and sigma annotation become base-row candidates; sequence-less rows are skipped with row-level provenance. |
| RegulonDB 11 `PromoterSet.csv` | Historical curated base source and release comparison | Rows are preserved as versioned source records and deduplicated by canonical sequence against RegulonDB 13. |
| Live RegulonDB 14.5 GraphQL `getAllOperon` promoter route | Modern route-completeness check and future overlay candidate | Current bounded probe showed the route is useful but incomplete, so live rows are not the present base source. |
| RegulonDB 13 sigmulon promoter CSVs | Supplemental sigma affiliation evidence | Inventoried/deferred for reconciliation; it does not create promoter base rows by itself. |
| RegulonDB 11 RACE and 454 promoter datasets | Experimental TSS/promoter evidence layer | Inventoried/deferred as experimental evidence, not curated base sequence truth. |
| RegulonDB 11 `PromoterPredictionSet.csv` | Computational prediction layer | Inventoried/deferred as predicted promoter candidates; prediction rows are not curated promoters and are not activity measurements. |
| EcoCyc 28 promoter SmartTable | Independent curation cross-check | Inventoried/deferred; EcoCyc windows can be longer than RegulonDB windows and must not replace RegulonDB sequences silently. |

The single USR dataset deduplicates canonical sequence records while preserving
source-specific promoter identities and affiliations in relation sidecars. Dense
`regulondb__*` base overlays summarize source strata, alias counts, canonical
sigma sets, confidence sets, and metadata completeness. Sparse or many-to-many
facts, including citations, individual regulators, boxes, TFBSs, prediction
scores, and source-specific raw columns, belong in relation sidecars. Sequence-
less curated source rows cannot become USR sequence base rows, and sequence-
bearing rows without sigma evidence are excluded from the strict base set; both
classes remain visible in provenance sidecars with source, release, table, row
reference, promoter identity, raw checksum, and reason.

Sequence-view identity is intentionally generic. Retained native RegulonDB rows
are exposed as `product_kind=source_record`, `context_kind=native_reference`,
`orientation=unknown`, and `recommended_pooling=seq_mean`. The mutable
`_views/view_semantics.parquet` addendum carries
`source_family=regulondb_native_promoter`,
`selection_basis=regulondb_curated_promoter_sequence_with_sigma`, and the
`regulondb_native_promoter_panel` collection tag. These rows are not
`analysis_window` products; core60 or context products must be derived later by
an explicit Construct/USR derivation step.

### Guardrails

- Cruncher owns RegulonDB source parsing and deterministic exports.
- USR imports only a fixed Cruncher export and defaults to dry-run.
- Base rows are sequence records using `bio_type=dna`; promoter ids stay in aliases and relation sidecars.
- Base-row inclusion is strict: records must have usable ACGT DNA sequence, stable promoter identity, retained provenance, and at least one sigma affiliation after sequence-level grouping.
- No duplicate canonical sequence rows are allowed in the committed USR dataset.
- Source promoter ids, names, sigma affiliations, predictions, and cross-release differences stay provenance-qualified instead of being flattened into sparse base columns.
- Sigma labels are summarized with canonical labels such as `sigma70`, while raw source labels remain in relation/source-row provenance.
- Native RegulonDB sigma metadata uses `regulondb__*` fields and never writes `sig35_variant`.
- Prediction rows are computationally inferred promoter candidates. They must not be interpreted as measured promoter activity or curated promoter strength.
- This study is separate from `stress_ethanol_cipro_growth`; comparisons require an explicit cross-study workspace.

### Current Row Counts

- `usr_regulondb_native_promoters`: 3,182 base rows (`local validated`, untracked)
- `usr_regulondb_native_promoters/_views/sequence_views.parquet`: 3,182 `source_record` views
- `usr_regulondb_native_promoters/_views/view_semantics.parquet`: 3,182 mutable semantics rows
- `infer_regulondb_native_promoter_views_7b`: 3,182 source-record
  `seq_mean` views complete; 6,364 vector keys and 6,364 scalar keys are
  reusable from canonical native `_derived/infer` sidecars
- `usr_regulondb_native_promoter_core60`: 3,181 canonical 60 bp sequence rows plus 3,182 `analysis_window` sequence views (`local validated`, generated/untracked)
- `usr_regulondb_native_promoters/_derived/infer`: 12,728 feature alias rows,
  12,728 feature vector rows, 6,364 scalar alias rows, and 6,364 scalar rows
  after the native/full plus in-context core60 dogfood run. The checked-in
  native/full runbook quota reuses the 6,364 `seq_mean` vector rows and 6,364
  scalar rows from this sidecar set.
- `usr_regulondb_native_promoter_core60/_derived/infer`: 6,364 feature alias
  rows and 6,364 scalar alias rows. The payload sidecars have 6,362 vector
  rows and 6,362 scalar rows because one duplicate 60 bp sequence is reused by
  two sequence views; alias rows preserve the view-level identities.

### TSS/Core60 Contract

The current local dataset has no populated -10/-35 box annotations:
`_relations/promoter_boxes.parquet` has zero rows, and
`regulondb__has_minus10_box` / `regulondb__has_minus35_box` are false for all
3,182 base records. It does have TSS coverage for all retained base records.
RegulonDB PromoterSet source sequences are 81 bp native windows with the
transcription start site at sequence offset `60`, so the configured Construct
core60 route emits the half-open upstream interval `[0,60)` as an
`analysis_window`. This is a TSS-upstream source-window contract, not inferred
box centering.

### Read-Only Probe Evidence

Latest local source probe on 2026-04-27 produced a temporary Cruncher superset
export outside the repo. The probe parsed the base-row-capable curated
PromoterSet sources from sibling `dnadesign-data` and inventoried supplemental
source strata without turning them into sequence rows:

- Normalized curated source records: 7,914 from RegulonDB 13 and RegulonDB 11 PromoterSet files.
- Skipped curated source rows with row-level provenance: 184 total, all `missing_sequence` (`92` from RegulonDB 13 and `92` from RegulonDB 11).
- Strict USR base rows after sigma-required canonical sequence deduplication: 3,182.
- Retained source rows: 6,629.
- Duplicate sequence collapses among retained rows: 3,447.
- Sequence-bearing source rows excluded for missing sigma: 1,285 source rows, representing 645 sequence groups.
- Same-release promoter-id sequence conflicts: 0.
- Supplemental strata recorded but deferred from base-row creation: RegulonDB 13 sigmulon, RegulonDB 11 RACE/454, RegulonDB 11 prediction rows, and EcoCyc 28 promoter windows.
- Local write-mode validation on 2026-04-28 created the single dataset layout with `records.parquet`, dense `regulondb__*` overlays, `_relations/*.parquet` provenance sidecars, and source-record sequence-view sidecars. The generated dataset path is ignored by git.

### Superset Fidelity Checks

The latest dry-run validation of the temporary superset reported:

- Duplicate canonical base sequences: 0.
- Invalid non-ACGT base sequences: 0.
- Base sequence length mismatches: 0.
- Required retained `regulondb__*` overlay metadata null counts: none.
- Orphan relation rows with `usr_id`: 0.
- Duplicate relation rows after alias and sigma deduplication: 0 for retained relation sidecars.
- Sigma labels after canonicalization: `sigma70` 3,991; `sigma24` 1,042; `sigma32` 624; `sigma38` 479; `sigma28` 290; `sigma54` 198; `sigma19` 1.
- Missing sigma annotation in retained base rows: 0. The 645 no-sigma sequence groups are omitted from `records.parquet` and preserved in `_relations/excluded_source_rows.parquet`.
- Fuzzy promoter-name collision candidates after strict filtering: 16. These are manual-review signals for name-based reconciliation, not automatic duplicate calls. Examples include suffix-near names such as `rsmDp1`/`rsmDp11`, `yhcAp1`/`yhcAp11`, and `sdiAp1`/`sdiAp1b`.

A bounded live RegulonDB 14.5 GraphQL probe on 2026-04-27 returned 20 promoter
records with 95% sequence coverage, 95% TSS coverage, and 55% sigma coverage.
The live route remains a modern overlay/completeness check rather than the
current completeness base.

### Current Downstream Posture

- Construct: materialized `usr_regulondb_native_promoter_core60` through
  `native_tss_upstream_core60` on 2026-04-29. The run wrote 3,182 sequence-view
  rows and 3,181 canonical 60 bp sequence rows; the difference is expected USR
  sequence deduplication for duplicate derived windows.
- Infer: the standard native source-record `seq_mean` and derived core60
  `core60_mean` Evo2 7B lanes are locally complete. Both lanes request
  intermediate block means, output-layer means, mean-per-token log likelihoods,
  and total log likelihoods. An additive dogfood config also extracts
  `core60_mean` over `[0,60)` from the full 81 bp native context in the same
  forward pass group as native `seq_mean`.
- Notify: native and core60 Infer event-path/profile smoke checks succeeded
  during local dogfood. Watcher cursors consumed the terminal Infer events with
  no remaining spool files.
- Ops: `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel`
  now discovers the native/full and core60 Infer runbooks, blocks stale or
  missing-product lanes, and plans only lanes with missing vectors/scalars. The
  2026-04-30 local plan marks both checked-in 7B lanes `skip_complete`, with
  zero blocked lanes, zero missing products, zero missing vectors, and zero
  missing scalars.
- LatentDNA: configured with a workspace and study binding. The workspace has
  validated with RegulonDB metadata cohorts and planned native/core60 feature
  source declarations. Refresh downstream snapshots/plots against the completed
  Infer sidecars before treating LatentDNA feature deliverables as current.
- Cluster: submit-ready runbooks exist, but no SCC submission is currently
  needed for the local RegulonDB 7B standard lanes because sidecars are
  complete.
- OPAL: not configured.

### End-to-End Readiness Audit

This study now has a maintained source-intake record, a tested USR import
script, a local validated native USR dataset, checked-in downstream contracts
for Construct, Infer, Notify event resolution, and LatentDNA planned feature
consumption, plus local Evo2 7B sidecars for the standard native/full and
derived core60 lanes.

Record-backed evidence through 2026-04-30:

- `ops progress show usr.data-plane.promoter-study-status --study-dir docs/studies/regulondb_native_promoter_panel --json` previously reported `is_active_study=false` and `native_full_infer_7b`; the checked-in record now advances the local phase to `local_infer_complete_7b` based on sidecar inventory.
- The same status surface reports `exists=true` and `rows=3182` for `usr_regulondb_native_promoters`, and reports the generated core60 dataset as present in the local canonical USR root.
- `uv run usr validate usr_regulondb_native_promoters --strict` passes locally.
- `uv run construct workspace run-project --workspace src/dnadesign/construct/workspaces/study_regulondb_native_promoter_panel --project native_tss_upstream_core60 --format json` returned `records_total=3182`, `records_written=3182`, and `dry_run=false`.
- `uv run usr validate usr_regulondb_native_promoter_core60 --strict` passes locally.
- `uv run infer validate config --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml` passes without GPU inventory.
- `uv run infer validate sequence-view-completion --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml --format json --mode inventory` now reports 3,182 required views with zero missing vectors and zero missing scalars.
- `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.native_full.evo2_7b.yaml --json` resolves the native USR `.events.log`.
- `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_regulondb_native_promoter_panel/config.sequence_views.core60_tss_upstream.evo2_7b.yaml --json` resolves the core60 USR `.events.log`.
- `uv run ops runbook fill-infer --study-dir docs/studies/regulondb_native_promoter_panel --repo-root . --plan-only` reports `lanes_total=2`, `skip_complete_lanes=2`, `runnable_lanes=0`, `blocked_lanes=0`, and zero missing or stale vector/scalar products.
- Local Evo2 7B dogfood on a Blackwell GPU completed the additive native/full
  plus in-context core60 config in 58.74 seconds wall time with about 18.8 GiB
  peak GPU memory observed, then completed the derived core60 standard lane in
  49.57 seconds wall time with about 17.9 GiB peak GPU memory observed.
- A read-only empirical recompute check sampled 128 native source records and
  128 derived core60 views through `Evo2Adapter('evo2_7b', 'cuda:0', 'bf16')`;
  fresh output-layer means, intermediate embedding means, and log-likelihood
  scalars matched persisted sidecars exactly (`max_abs_diff=0.0`, missing
  payloads `0`).
- A direct USR congruence check reports native records are all 81 bp, derived
  core60 records/views are all 60 bp, every core60 view has a parent native
  sequence, and `core60_sequence == parent_native_sequence[0:60]` for all
  3,182 views.
- `uv run ops progress show usr.data-plane.promoter-study-preflight --study-dir docs/studies/regulondb_native_promoter_panel --scope next --command-timeout-seconds 30 --json` reports `state=ok`, 13 ok checks, no missing checks, and no blockers when run from this checkout on 2026-04-30.
- `MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna validate workspace --workspace regulondb_native_promoter_panel --deep --json` returns `status=ok`.
- `MPLCONFIGDIR=/tmp/dnadesign_mpl uv run latentdna workspace snapshot --workspace regulondb_native_promoter_panel --json` writes the local partial snapshot contract. It is valid for native metadata review; feature plots should be refreshed against the completed sidecars when the downstream analysis step is resumed.

Residual operational footgun:

- Directly rerunning a completed Infer config is GPU-free but still spends about
  24.5 seconds and about 3.6 GiB RSS reconciling/re-writing completion
  sidecars. The study-level `fill-infer` path avoids this by skipping complete
  lanes; future optimization should add a fast no-op return inside direct
  `infer run` when all vector/scalar sidecars are already complete.

### Next Actions

- Review or sync the local generated USR datasets and Infer sidecars through the
  normal USR data sync path. Do not add generated dataset roots to git.
- Do not submit another standard RegulonDB Infer batch from this checkout unless
  completion inventory reports missing or stale vectors/scalars. The current
  handoff command is `uv run ops runbook fill-infer --study-dir
  docs/studies/regulondb_native_promoter_panel --repo-root . --plan-only`.
- Decide whether the additive full-native-context `[0,60)` pooled core60 values
  should become part of the official study quota. If yes, promote the additive
  config into a checked-in runbook; until then it remains dogfood evidence and
  a reusable config, not a default batch lane.
- Refresh LatentDNA snapshots and plots against the completed native/full and
  core60 sidecars.
- Keep live RegulonDB 14.5, sigmulon, HT, prediction, and EcoCyc strata as explicit future reconciliation work rather than silently widening the current base table.
