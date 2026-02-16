## OPAL Selection Strategies

Selection plugins convert selected score/uncertainty channels into ranking decisions.

## Strategy comparison

| Strategy | Inputs | Behavior | Typical pairing |
| --- | --- | --- | --- |
| `top_n` | `score_ref` | Deterministic rank-by-score | RF + SFXI, GP + SFXI |
| `expected_improvement` | `score_ref` + `uncertainty_ref` | Acquisition ranking (exploration/exploitation) | GP + SFXI |

## Runtime contract

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

## Config contract (v2)

Every selection config must include:

- `top_k`
- `score_ref`
- `objective_mode`
- `tie_handling`

`expected_improvement` additionally requires:

- `uncertainty_ref`
- The referenced uncertainty channel must be a standard deviation (not variance).

## Built-ins

### `top_n`

Deterministic ranking by selected score channel.

### `expected_improvement`

Uncertainty-aware acquisition ranking.

- consumes selected score channel (`score_ref`)
- consumes uncertainty standard deviation channel (`uncertainty_ref`)
- hard-fails on missing/non-finite/negative/all-zero uncertainty
- does not degrade to score-only behavior

Acquisition math:

- maximize mode: `I = s - s*` where `s* = max(scores)`
- minimize mode: `I = s* - s` where `s* = min(scores)`
- `Z = I / sigma`
- `EI = I * Phi(Z) + sigma * phi(Z)`
- OPAL selection score: `A = alpha * (I * Phi(Z)) + beta * (sigma * phi(Z))`

Where:

- `sigma` is the uncertainty standard deviation channel from `uncertainty_ref`
- `Phi` is the standard normal CDF
- `phi` is the standard normal PDF

Zero-`sigma` handling:

- Mixed per-candidate zero values are allowed.
- For each candidate with `sigma == 0`, OPAL uses the deterministic EI limit:
  - `A = alpha * max(I, 0)`
- If all candidates have `sigma == 0`, OPAL fails fast because the acquisition has no exploration signal.

Weighted-acquisition note:

- With `alpha != beta`, `A` can be negative for some candidates.
- This is valid and still rankable; OPAL only requires finite acquisition values.

## Example configs

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
