## Expected Improvement Plugin (`expected_improvement`)

This page documents `expected_improvement` acquisition behavior, equations, and failure conditions.
For registry-level selection contracts and required fields, see [Selection](./selection.md).

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

- `A = alpha * (I * Phi(Z)) + beta * (sigma * phi(Z))`

### Zero-sigma behavior

For per-candidate `sigma == 0`, OPAL uses the deterministic limit:

- `A = alpha * max(I, 0)`

This avoids `0/0` numerical failure in mixed-sigma batches.

If all candidates have `sigma == 0`, OPAL raises an error (no exploration signal).

### Error cases

`expected_improvement` errors on:

- missing `uncertainty_ref`
- non-finite uncertainty values
- negative uncertainty values
- all-zero uncertainty vector
- non-finite acquisition outputs after computation

There is no fallback to `top_n`.

### Practical pairing in OPAL

Most common path:

- model: `gaussian_process`
- objective: `sfxi_v1`
- selection: `expected_improvement`

### See also

- [Selection plugins](./selection.md)
- [Gaussian Process behavior and math](./model-gaussian-process.md)
- [SFXI behavior and math](./objective-sfxi.md)
- [GP + SFXI + expected_improvement workflow](../workflows/gp-sfxi-ei.md)
