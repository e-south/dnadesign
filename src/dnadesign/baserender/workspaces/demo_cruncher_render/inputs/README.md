# Cruncher Render Demo Inputs

This workspace is the single Cruncher-focused baserender demo.

It keeps both shapes needed for practical iteration:

1. Source-like Cruncher artifacts (runtime-adjacent mock inputs):
   - `elites.parquet`
   - `elites_hits.parquet`
   - `config_used.yaml`

2. Hotpath baserender Record-shape input (fast visual iteration):
   - `elites_showcase_records.parquet`

`elites_showcase_records.parquet` contains one row per elite with the normalized columns consumed by the `generic_features` adapter:

- `id`
- `sequence`
- `features`
- `effects`
- `display`

The workspace `job.yaml` is wired to the hotpath record-shape file so design iteration stays fast, while the source-like artifacts remain available for contract checks and future adapter regeneration.
