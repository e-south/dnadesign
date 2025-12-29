## OPAL Transform Y Strategies

Transforms convert raw measurement tables into the
campaign’s canonical **Y** label **per design**.

#### Contract

A transform is a **pure function** registered in the Y‑transform registry:

```python
def transform_fn(df_tidy: pd.DataFrame, params: dict, ctx: PluginCtx | None) -> pd.DataFrame
````

#### Input:

* `df_tidy`: raw measurements (e.g., design or campaign-specific).
* `params`: transform-specific schema and pre-processing controls (from YAML).
* `ctx`: RoundCtx plugin context (for contract enforcement/audit).

#### Output:

* DataFrame with at least columns:

- `id` (design_id), optional; when absent, OPAL resolves by `sequence`.
- `y` (list[float]) — the canonical label for this campaign (vector).

Transforms must fail fast (raise `OpalError`) on schema or invariant violations
(duplicate timepoints, missing states/channels, NaNs/Inf, wrong lengths).

#### Y‑ops (training-time transforms)

Y‑ops are registered via `register_y_op(...)` and must provide
`fit/transform/inverse` functions. OPAL enforces contracts and records their
outputs under `yops/<name>/...` in `round_ctx.json`.
