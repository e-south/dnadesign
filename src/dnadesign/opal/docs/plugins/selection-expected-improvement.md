## Expected Improvement Plugin (`expected_improvement`)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This page documents `expected_improvement` acquisition behavior, equations, and failure conditions. For registry-level selection contracts and required fields, see [Selection](./selection.md).

### Purpose

`expected_improvement` ranks candidates by balancing:

- exploitation (high predicted score), and
- exploration (high predictive uncertainty standard deviation).

### Inputs and channel refs

Configured in `selection.params`:

- `score_ref`: `<objective>/<score_channel>`
- `uncertainty_ref`: `<objective>/<uncertainty_channel>` (required for EI)
- `objective_mode`: `maximize|minimize`
- `top_k`, `tie_handling`, optional `alpha`, `beta`

`uncertainty_ref` must resolve to standard deviation values in the same objective units as `score_ref`.

### Wiring patterns (important)

The `<objective>/<channel>` ref identifies a **channel key** within that objective. Score and uncertainty are separate surfaces:

* `score_ref` pulls the **score values** for that channel key.
* `uncertainty_ref` pulls the **standard deviation values** for that channel key.

Some objectives publish uncertainty under the **same channel key** as the score (SFXI does this for `sfxi`). In that case it is valid for `score_ref` and `uncertainty_ref` to be identical.

Minimal example (SFXI + EI):

```yaml
selection:
  name: expected_improvement
  params:
    top_k: 5
    score_ref: sfxi_v1/sfxi
    uncertainty_ref: sfxi_v1/sfxi
    objective_mode: maximize
    alpha: 1.0
    beta: 1.0
```

Common pitfall: setting `uncertainty_ref` to a channel key that **does not** publish uncertainty (or running EI with a model/objective path that produces no uncertainty). OPAL fails fast and does not fall back to `top_n`.

### Acquisition math

Let:

- `s` = selected score channel value for a candidate
- `s*` = current best score under the configured objective mode
- `I` = improvement term
- `sigma` = uncertainty standard deviation
- `Phi` = standard normal CDF
- `phi` = standard normal PDF

Improvement:

- maximize: `I = s - s*`
- minimize: `I = s* - s`

Standard EI:

- `Z = I / sigma`
- `EI = I * Phi(Z) + sigma * phi(Z)`

OPAL weighted acquisition:

- `sigma_norm = (sigma - min(sigma)) / (max(sigma) - min(sigma))` (clipped to `[0,1]`; all-equal sigma yields zeros)
- `A = alpha * (I * Phi(Z)) + beta * (sigma_norm * phi(Z))`

Important:

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
