## OPAL X Transforms

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This page documents X-transform plugin contracts and runtime expectations. Use it when configuring `transforms_x` in `campaign.yaml` or implementing a new X transform plugin.

**Contract**

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

### Runtime carrier contracts

X transforms may declare `@roundctx_contract(category="transform_x", ...)` on the factory to
enforce and audit their runtime keys in `round_ctx.json`.
