## RegulonDB Source Model

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-04

This page owns durable source and ontology semantics for the
`regulondb_native_promoter_panel` study. Keep current phase, row counts, and
next actions in `../../record/status.md`.

### Placement Boundary

This study is a data intake and validation lane, not a new top-level
dnadesign tool. Source file discovery uses the sibling `dnadesign-data` public
API. Cruncher normalizes source rows into a fixed export. USR turns that fixed
export into one sequence-level dataset after validation passes. Generated
exports, USR parquet sidecars, snapshots, and downstream embeddings stay out of
the checked-in study record.

### Provenance Model

`usr_regulondb_native_promoters` is the single USR dataset for the strict
native promoter superset. It is not a live-only dataset. Its input is a fixed
Cruncher superset export that records source, release, table or API route, row
reference, raw checksum, parser version, and source stratum before USR import.

| Stratum | Role | Provenance rule |
| --- | --- | --- |
| RegulonDB 13 `PromoterSet.tsv` | Current curated release-pinned base source | Strict DNA sequence and sigma annotation become base-row candidates; sequence-less rows remain skipped provenance. |
| RegulonDB 11 `PromoterSet.csv` | Historical curated base source and release comparison | Rows are preserved as versioned source records and deduplicated by canonical sequence against RegulonDB 13. |
| Live RegulonDB 14.5 GraphQL `getAllOperon` promoter route | Modern route-completeness check and future overlay candidate | Useful but incomplete in the bounded probe, so live rows are not the present base source. |
| RegulonDB 13 sigmulon promoter CSVs | Supplemental sigma affiliation evidence | Inventoried/deferred for reconciliation; does not create promoter base rows by itself. |
| RegulonDB 11 RACE and 454 promoter datasets | Experimental TSS/promoter evidence layer | Inventoried/deferred as experimental evidence, not curated base sequence truth. |
| RegulonDB 11 `PromoterPredictionSet.csv` | Computational prediction layer | Inventoried/deferred; prediction rows are not curated promoters and are not activity measurements. |
| EcoCyc 28 promoter SmartTable | Independent curation cross-check | Inventoried/deferred; EcoCyc windows must not silently replace RegulonDB sequences. |

The single USR dataset deduplicates canonical sequence records while preserving
source-specific promoter identities and affiliations in relation sidecars.
Dense `regulondb__*` base overlays summarize source strata, alias counts,
canonical sigma sets, confidence sets, and metadata completeness. Sparse or
many-to-many facts belong in relation sidecars.

Sequence-view identity is intentionally generic. Retained native RegulonDB rows
are exposed as `product_kind=source_record`, `context_kind=native_reference`,
`orientation=unknown`, and `recommended_pooling=seq_mean`. Mutable
`_views/view_semantics.parquet` rows carry source-family, selection-basis, and
collection tags.

### Guardrails

- Cruncher owns RegulonDB source parsing and deterministic exports.
- USR imports only a fixed Cruncher export and defaults to dry-run.
- Base rows are sequence records using `bio_type=dna`; promoter ids stay in
  aliases and relation sidecars.
- Base-row inclusion is strict: records must have usable ACGT DNA sequence,
  stable promoter identity, retained provenance, and at least one sigma
  affiliation after sequence-level grouping.
- Source promoter ids, names, sigma affiliations, predictions, and
  cross-release differences stay provenance-qualified instead of being
  flattened into sparse base columns.
- Native RegulonDB sigma metadata uses `regulondb__*` fields and never writes
  `sig35_variant`.
- Prediction rows are computationally inferred promoter candidates. They must
  not be interpreted as measured promoter activity or curated promoter
  strength.
- This study is separate from `stress_ethanol_cipro_growth`; comparisons
  require an explicit cross-study workspace.

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
