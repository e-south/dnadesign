# DenseGen Integration Contract

This page defines how DenseGen record outputs map into `baserender`.

## Contract intent

DenseGen remains the producer of run artifacts and metadata.
`baserender` is only responsible for adapting DenseGen rows into `Record` and rendering figures.

## Adapter contract

Default adapter kind: `densegen_tfbs`

Default mapping used by DenseGen notebook scaffolding:

- `sequence` -> `sequence`
- `annotations` -> `densegen__used_tfbs_detail`
- `id` -> `id`

Required source columns for the default mapping:
- `id`
- `sequence`
- `densegen__used_tfbs_detail`
- Fixed style preset from DenseGen notebook contract: `presentation_default`
- DenseGen notebook preview window limit: `500` rows

The adapter contract supports an optional `overlay_text` column key for notebook/UI overlays; DenseGen notebook scaffolding does not set it and reads only the records contract mapping above.

## Public API flow

Use public package imports only:

```python
from dnadesign.baserender import load_records_from_parquet, render_record_figure
from dnadesign.densegen import densegen_notebook_render_contract

contract = densegen_notebook_render_contract()
records = load_records_from_parquet(
    dataset_path="/path/to/records.parquet",
    record_ids=["record_id"],
    adapter_kind=contract.adapter_kind,
    adapter_columns=contract.adapter_columns,
    adapter_policies=contract.adapter_policies,
)
fig = render_record_figure(records[0], style_preset=contract.style_preset)
```

Do not import from `dnadesign.baserender.src.*`.

## DenseGen notebook boundary

`dense notebook generate` resolves one records parquet source from DenseGen output wiring:
- single sink -> that sink (`parquet` or `usr`)
- multiple sinks -> `plots.source`
The scaffolded notebook then uses BaseRender public API helpers (`load_records_from_parquet`, `render_record_figure`) with the DenseGen contract module.

## Demo workspace

Validate and run the curated DenseGen demo workspace:

```bash
uv run baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
```

The demo parquet row includes the same nested DenseGen lineage-style fields (`motif_id`, `tfbs_id`, stage-A metadata, strand/offset fields) seen in real DenseGen outputs.
