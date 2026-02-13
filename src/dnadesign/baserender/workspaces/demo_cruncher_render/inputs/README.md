# Cruncher Render Demo Inputs

This workspace is the single Cruncher-focused baserender demo.

It keeps both shapes needed for practical iteration:

1. Source-like Cruncher artifacts (runtime-adjacent mock inputs):
   - `elites.parquet`
   - `elites_hits.parquet`
   - `config_used.yaml`
   - `lockfile.json`
   - `motif_store/normalized/motifs/demo_merged_meme_oops/*.json`

2. Hotpath baserender Record-shape input (fast visual iteration):
   - `elites_showcase_records.parquet`

`elites_showcase_records.parquet` contains one row per elite with the normalized columns consumed by the `generic_features` adapter:

- `id`
- `sequence`
- `features`
- `effects`
- `display`

The workspace `job.yaml` is wired to the hotpath record-shape file so design iteration stays fast, while the source-like artifacts remain available for contract checks and future adapter regeneration.

The demo job enables `attach_motifs_from_cruncher_lockfile` so motif logos are sourced from lockfile-resolved catalog motif records (the same MEME/OOPS-derived motif artifacts used by Cruncher), never inferred from elite windows.

Checksum fields in the demo `lockfile.json` and motif records are sanitized placeholders for repository safety checks; the transform still enforces strict lockfile-to-motif checksum equality.
