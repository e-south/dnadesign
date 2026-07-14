---
id: opal-selection-plugins
title: OPAL selection plugins
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-14
---

## OPAL Selection Strategies

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14


Selection plugins consume named objective channels and emit ranked candidate
sets under the contracts below.

### Built-in strategies

| Strategy | Inputs | Behavior | Typical pairing |
| --- | --- | --- | --- |
| `top_n` | `score_ref` | Deterministic rank-by-score | Any model and scalar objective channel |
| `expected_improvement` | `score_ref` + `uncertainty_ref` | Acquisition ranking (exploration/exploitation) | A model/objective pair that emits scalar uncertainty |

Source modules:

- `src/dnadesign/opal/src/selection/top_n.py`
- `src/dnadesign/opal/src/selection/expected_improvement.py`

### Selection detail pages

- [Expected Improvement behavior and math](expected-improvement.md)

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
Selection plugins do not inspect assay vectors or objective parameters; they
consume only the configured score and uncertainty channels.

### Config contract (v3)

Every `selection_views[].selection` config must include:

- `top_k`
- `score_ref`
- `objective_mode`
- `tie_handling`

Set `require_exact_top_k: true` when the study requires exactly `top_k`
candidates. A boundary tie under `competition_rank` or `dense_rank` then stops
the round with an explicit cardinality error. OPAL does not silently truncate a
tie or fill a short selection. The default is `false` because some campaigns
intentionally preserve all tied candidates.

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
- Acquisition details: [Expected Improvement behavior and math](expected-improvement.md)

### Example configs

Top-N:

```yaml
selection_views:
  - id: primary
    objective: {name: scalar_identity_v1, params: {}}
    selection:
      name: top_n
      params:
        top_k: 12
        score_ref: scalar
        objective_mode: maximize
        tie_handling: competition_rank
        require_exact_top_k: true
```

Expected improvement:

```yaml
selection_views:
  - id: primary
    objective: {name: sfxi_v1, params: {...}}
    selection:
      name: expected_improvement
      params:
        top_k: 12
        score_ref: sfxi
        uncertainty_ref: sfxi
        objective_mode: maximize
        tie_handling: competition_rank
        alpha: 1.0
        beta: 1.0
```

### See also

- [Model plugins](../models/README.md)
- [Gaussian Process behavior and math](../models/gaussian-process.md)
- [Objective plugins](../objectives/README.md)
- [GP + expected_improvement workflow](../../workflows/gp-sfxi-ei.md)
