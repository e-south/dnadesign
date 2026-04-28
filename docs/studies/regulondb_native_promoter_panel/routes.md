## regulondb_native_promoter_panel Routes

**Last verified:** 2026-04-28

Use this page after the checked-in study status establishes the current phase.
This study remains inactive/draft for downstream work, but the source dataset
has been generated locally for validation. The generated USR dataset root is not
git-tracked.

- Status: `uv run ops progress show usr.data-plane.promoter-study-status --study-dir docs/studies/regulondb_native_promoter_panel --json`
- Preflight: `uv run ops progress show usr.data-plane.promoter-study-preflight --study-dir docs/studies/regulondb_native_promoter_panel --scope next --json`

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
- Current state: `planned`
- Entry artifact: `usr_regulondb_native_promoters`
- Exit artifact: `infer_regulondb_native_promoter_views_7b`
- Route note: This route remains blocked until USR validation passes.

### LatentDNA Native Audit

- Type: `route`
- Plane: `data-plane`
- Surface role: `downstream-analysis`
- Owner-boundary: `latentdna`
- Current state: `planned`
- Entry artifact: native/full and later core60 7B feature surfaces
- Route note: Native cohorts use `regulondb__*` fields. They must not derive DenseGen metadata or alias native sigma factors into `sig35_variant`.
