## USR Import Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](README.md).

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
