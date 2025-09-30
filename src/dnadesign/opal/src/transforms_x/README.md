## OPAL X‑Transforms

X‑transforms convert your stored representation column (**X**) into a fixed‑width numeric matrix for models.

**Contract**

Registered under `registries/transforms_x.py`, a factory returns a callable:

```python
def factory(params: dict) -> Callable[[pd.Series], np.ndarray]:
    ...
```

The callable:

* accepts a **Series** of per‑record X values (scalar, list, JSON string, etc.),
* returns an `np.ndarray` of shape `(N, F)` with finite floats,
* raises with a clear message on mismatch (e.g., inconsistent widths).

Example: `identity` transforms scalars and vectors as‑is, coercing to `(N, F)` and enforcing finiteness.
