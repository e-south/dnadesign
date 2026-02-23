## OPAL Selection Strategies

This page documents selection plugin contracts, required config fields, and runtime output expectations.

### Built-in strategies

| Strategy | Inputs | Behavior | Typical pairing |
| --- | --- | --- | --- |
| `top_n` | `score_ref` | Deterministic rank-by-score | RF + SFXI, GP + SFXI |
| `expected_improvement` | `score_ref` + `uncertainty_ref` | Acquisition ranking (exploration/exploitation) | GP + SFXI |

Source modules:

- `src/dnadesign/opal/src/selection/top_n.py`
- `src/dnadesign/opal/src/selection/expected_improvement.py`

### Selection detail pages

- [Expected Improvement behavior and math](./selection-expected-improvement.md)

### Runtime contract

```python
def selection_fn(
    *,
    ids: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    objective: str,
    tie_handling: str,
    scalar_uncertainty: np.ndarray | None = None,
    ctx: PluginCtx | None = None,
    **plugin_params,
) -> dict
```

Required outputs:

- `order_idx`: sorted candidate indices as an integral permutation of `[0..n-1]`
- `score`: numeric selection score vector (finite, length `n`) used for writeback/verification

OPAL validates selection output types/shapes/finiteness before writeback.
Tie expansion (`top_k` with `competition_rank` or `dense_rank`) is computed from the plugin-returned `score` vector.

### Config contract (v2)

Every selection config must include:

- `top_k`
- `score_ref`
- `objective_mode`
- `tie_handling`

`expected_improvement` additionally requires:

- `uncertainty_ref`
- The referenced uncertainty channel must be a standard deviation (not variance).

### Built-ins

### `top_n`

Deterministic ranking by selected score channel.

### `expected_improvement`

Uncertainty-aware acquisition ranking.

- consumes selected score channel (`score_ref`)
- consumes uncertainty standard deviation channel (`uncertainty_ref`)
- ranks by EI score first, then predicted score (objective-aware), then `id`
- raises an error on missing/non-finite/non-positive uncertainty
- does not degrade to score-only behavior
- Acquisition details: [Expected Improvement behavior and math](./selection-expected-improvement.md)

### Example configs

Top-N:

```yaml
selection:
  name: top_n
  params:
    top_k: 12
    score_ref: "scalar_identity_v1/scalar"
    objective_mode: maximize
    tie_handling: competition_rank
```

Expected improvement:

```yaml
selection:
  name: expected_improvement
  params:
    top_k: 12
    score_ref: "sfxi_v1/sfxi"
    uncertainty_ref: "sfxi_v1/sfxi"
    objective_mode: maximize
    tie_handling: competition_rank
    alpha: 1.0
    beta: 1.0
```

### See also

- [Model plugins](./models.md)
- [Gaussian Process behavior and math](./model-gaussian-process.md)
- [Objective plugins](./objectives.md)
- [GP + expected_improvement workflow](../workflows/gp-sfxi-ei.md)
