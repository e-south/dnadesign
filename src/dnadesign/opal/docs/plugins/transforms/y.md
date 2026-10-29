<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[OPAL docs](../../index.md) / [Data transforms](README.md)

# Measurement and training transforms

Use `transforms_y` to map measurement-table columns into each candidate's outcome
vector. Use `training.y_ops` for reversible transformations applied during model
fitting. OPAL converts predictions back to the outcome units before the objective
scores them.

## Contract

A transform is a pure function registered in the Y-transform registry. Its
interface is:

```python
def transform_fn(df_tidy: pd.DataFrame, params: dict, ctx: PluginCtx | None) -> pd.DataFrame:
    ...
```

### Input

* `df_tidy`: raw measurements (e.g., design or campaign-specific).
* `params`: transform-specific schema and pre-processing controls (from YAML).
* `ctx`: RoundCtx plugin context (for contract enforcement/audit).

### Output

The returned DataFrame contains `y` as a list of floats for each observation.
It may include `id`; otherwise OPAL matches observations by `sequence`.

Transforms must fail fast on schema or invariant violations
(duplicate timepoints, missing states/channels, NaNs/Inf, wrong lengths).
Raising `OpalError` is preferred; other exceptions are normalized to `OpalError`
by the ingest pipeline with a friendly hint.

## Y‑ops (training-time transforms)

Y‑ops are registered via `register_y_op(...)` and must provide
`fit/transform/inverse` functions. OPAL enforces contracts and records their
outputs under `yops/<name>/...` in `round_ctx.json`.

## Built-in Y transforms

* `sfxi_vec8_from_table_v1` — 8‑vector (logic + intensity) from tidy SFXI table.
* `scalar_from_table_v1` — scalar Y from a single column (`y` by default).
* `vector_from_table_v1` — vector Y from an explicitly ordered list of columns.
