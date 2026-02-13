# Cruncher Showcase Job Contract

This contract is strict and explicit. Unknown keys fail.

## Top-level keys
- `version` (must be `3`)
- `results_root` (optional)
- `input`
- `selection` (optional)
- `pipeline` (optional)
- `render`
- `outputs` (required, non-empty list)
- `run` (optional)

### `results_root` default
- For non-workspace jobs: `<caller_root>/results` where `caller_root` defaults to current working directory.
- For workspace jobs loaded from `job.yaml` where sibling `inputs/` and `outputs/` exist:
  `<workspace>/outputs`

## Input
- `kind: parquet`
- `path: str`
- `adapter`
  - `kind: densegen_tfbs | generic_features | cruncher_best_window`
  - `columns: mapping` (adapter-specific, strict)
  - `policies: mapping` (adapter-specific, strict, unknown keys fail)
- `alphabet: DNA|RNA|PROTEIN`
- `limit: int|null`
- `sample: { mode: first_n|random_rows, n: int, seed: int|null } | null`

## Selection
- `path, match_on, column, overlay_column, keep_order, on_missing`
- If `overlay_column` is set and missing in CSV, validation fails.
- Selection always filters to keys present in the CSV.
- Selection rejects duplicate record keys for the selected `match_on` field.
- `keep_order: true` preserves CSV order (including duplicate keys).
- `keep_order: false` keeps only selected keys and uses deterministic key order:
  - `id` / `sequence`: lexicographic
  - `row`: numeric

## Render
- `renderer: sequence_rows`
- `style: { preset: str|path|null, overrides: mapping }`

## Outputs
Each item is one of:
- `kind: images`
  - `dir: str|null`
  - `fmt: png|svg|pdf`
- `kind: video`
  - `path: str|null`
  - `fmt: mp4`
  - `fps >= 1`
  - `frames_per_record >= 1`
  - optional `pauses, width_px, height_px, aspect, total_duration`

No implicit outputs: only declared outputs run.

Default output paths:
- Non-workspace jobs: `<results_root>/<job_stem>/...`
- Workspace jobs (default `results_root=<workspace>/outputs`): `<workspace>/outputs/...`

## Run
- `strict`
- `fail_on_skips`
- `emit_report`
- `report_path`
