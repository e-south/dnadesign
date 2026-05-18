## Source Intake Route Detail

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-18

Parent router: [README.md](../README.md).

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
