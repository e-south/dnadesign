## OPAL Y Transforms

This page documents Y-transform and y-op contracts used during ingest and model-time Y processing. Use it when configuring `transforms_y` for ingest and `training.y_ops` for runtime inversion behavior.

### Contract

A transform is a **pure function** registered in the Y‑transform registry:

```python
def transform_fn(df_tidy: pd.DataFrame, params: dict, ctx: PluginCtx | None) -> pd.DataFrame
```

### Input:

* `df_tidy`: raw measurements (e.g., design or campaign-specific).
* `params`: transform-specific schema and pre-processing controls (from YAML).
* `ctx`: RoundCtx plugin context (for contract enforcement/audit).

### Output:

* DataFrame with at least columns:

- `id` (design_id), optional; when absent, OPAL resolves by `sequence`.
- `y` (list[float]) — the primary label for this campaign (vector).

Transforms must fail fast on schema or invariant violations
(duplicate timepoints, missing states/channels, NaNs/Inf, wrong lengths).
Raising `OpalError` is preferred; other exceptions are normalized to `OpalError`
by the ingest pipeline with a friendly hint.

### Y‑ops (training-time transforms)

Y‑ops are registered via `register_y_op(...)` and must provide
`fit/transform/inverse` functions. OPAL enforces contracts and records their
outputs under `yops/<name>/...` in `round_ctx.json`.

### Built-in Y transforms

* `sfxi_vec8_from_table_v1` — 8‑vector (logic + intensity) from tidy SFXI table.
* `scalar_from_table_v1` — scalar Y from a single column (`y` by default).
