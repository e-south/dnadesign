## Pool-relative weighted acquisition (`expected_improvement`)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-19


`expected_improvement` is the stable registry identifier. The implementation is
a pool-relative weighted acquisition heuristic, not classical expected
improvement against the best observed outcome. It follows the equations and
failure conditions below.
Registry-level contracts and required fields are listed under
[Selection](README.md).

### Purpose

The plugin ranks candidates by balancing:

- exploitation (high predicted score), and
- exploration (high predictive uncertainty standard deviation).

### Inputs and channel refs

Configured in `selection_views[].selection.params`:

- `score_ref`: objective-local score channel
- `uncertainty_ref`: objective-local uncertainty channel (required for EI)
- `objective_mode`: `maximize|minimize`
- `top_k`, `tie_handling`, optional `alpha`, `beta`

`uncertainty_ref` must resolve to standard deviation values in the same objective units as `score_ref`.

### Wiring patterns (important)

Refs identify channel keys emitted by the objective in the same named view.
Persisted refs are namespaced by view ID. Score and uncertainty are separate
surfaces:

* `score_ref` pulls the **score values** for that channel key.
* `uncertainty_ref` pulls the **standard deviation values** for that channel key.

Some objectives publish uncertainty under the **same channel key** as the score (SFXI does this for `sfxi`). In that case it is valid for `score_ref` and `uncertainty_ref` to be identical.

Minimal v3 example (SFXI + EI):

```yaml
selection_views:
  - id: primary
    objective: {name: sfxi_v1, params: {...}}
    selection:
      name: expected_improvement
      params:
        top_k: 5
        score_ref: sfxi
        uncertainty_ref: sfxi
        objective_mode: maximize
        tie_handling: competition_rank
        alpha: 1.0
        beta: 1.0
```

Common pitfall: setting `uncertainty_ref` to a channel key that **does not** publish uncertainty (or running EI with a model/objective path that produces no uncertainty). OPAL fails fast and does not fall back to `top_n`.

### Acquisition math

Let:

- `s` = selected score channel value for a candidate
- `s*` = best predicted score in the current candidate pool under the
  configured objective mode
- `I` = improvement term
- `sigma` = uncertainty standard deviation
- `Phi` = standard normal CDF
- `phi` = standard normal PDF

Improvement:

- maximize: `I = s - s*`
- minimize: `I = s* - s`

EI-shaped terms:

- `Z = I / sigma`
- `EI = I * Phi(Z) + sigma * phi(Z)`

OPAL weighted acquisition:

- `sigma_norm = (sigma - min(sigma)) / (max(sigma) - min(sigma))` (clipped to `[0,1]`; all-equal sigma yields zeros)
- `A = alpha * (I * Phi(Z)) + beta * (sigma_norm * phi(Z))`

Important:

- the incumbent is pool-relative and predicted, not the best observed outcome;
- changing candidate-pool membership can change the incumbent and the
  min-max-normalized uncertainty term;
- the exploitation term remains in score units while the normalized exploration
  term is dimensionless, so `alpha` and `beta` also define an implicit unit
  conversion; rescaling the score can change ranks;
- raw `sigma` is used in `Z = I / sigma` (no sigma normalization in z-score denominator)
- only the exploration multiplier uses `sigma_norm`

Normalization:

- OPAL min-max normalizes weighted acquisition to `[0,1]` before returning `score`.
- `A_norm = (A - min(A)) / (max(A) - min(A))`
- if `max(A) == min(A)`, OPAL returns all zeros.

Ranking source:

- OPAL ranks by `A_norm` (descending).
- If `A_norm` ties, OPAL breaks ties by predicted score:
  - `maximize`: higher predicted score first
  - `minimize`: lower predicted score first
- If both `A_norm` and predicted score tie, ranking is resolved deterministically by candidate `id`.

### Sigma contract

EI requires strict positive standard deviation for every candidate:

- `sigma > 0` for all candidates

Any non-positive value (`sigma <= 0`) is an error; there is no epsilon-tolerance override.

### Error cases

`expected_improvement` errors on:

- missing `uncertainty_ref`
- non-finite uncertainty values
- non-positive uncertainty values
- non-finite acquisition outputs after computation

There is no fallback to `top_n`.
