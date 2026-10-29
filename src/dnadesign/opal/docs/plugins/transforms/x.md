<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[OPAL docs](../../index.md) / [Data transforms](README.md)

# Prepare candidate features

Feature transforms turn per-candidate vectors into the matrix used by the
predictor. For existing numeric feature vectors, configure the built-in
`identity` transform:

```yaml
# Use the vectors in data.x_column_name without changing their values.
transforms_x: {name: identity, params: {}}
```

The resulting matrix has one row per candidate and one column per feature.
See [candidate tables](../../reference/data-contracts.md#prepare-a-candidate-table)
for the input format.

## Implement a feature transform

Registered under `registries/transforms_x.py`, a factory returns a callable:

```python
def factory(params: dict) -> Callable[[pd.Series, PluginCtx | None], np.ndarray]:
    ...
```

The callable:

* accepts a **Series** of per-record vector cells from the canonical X column,
* accepts a `ctx` (RoundCtx plugin context) for contract enforcement/audit,
* returns an `np.ndarray` of shape `(N, F)` with finite floats,
* raises with a clear message on mismatch (e.g., inconsistent widths).

Example: `identity` passes vector cells through to `(N, F)` and enforces
finiteness. Scalar cells and JSON-string vectors are import/normalization
inputs, not runtime campaign inputs.

## Runtime carrier contracts

X transforms may declare `@roundctx_contract(category="transform_x", ...)` on the factory to
enforce and audit their runtime keys in `round_ctx.json`.
