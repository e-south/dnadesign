# Cruncher Render Demo Inputs

This workspace is the single Cruncher-focused baserender demo.

Only runtime-essential inputs are kept (YAGNI):

- `elites_showcase_records.parquet`
- `motif_library.json`

`elites_showcase_records.parquet` contains one row per elite with the normalized columns consumed by the `generic_features` adapter:

- `id`
- `sequence`
- `features`
- `effects`
- `display`

The workspace `job.yaml` is wired to this record-shape hotpath for fast iteration.

The demo job uses `attach_motifs_from_library` so baserender consumes a simple tool-provided motif primitives contract (`motif_library.json`) instead of depending on Cruncher catalog paths.

`motif_library.json` is the canonical motif source for this demo, so logos are sourced from optimization motifs, not inferred from elite windows.
