## Gaussian Process Plugin (`gaussian_process`)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-08


`gaussian_process` follows the equations and runtime invariants below.
Registry-level contracts and model inventory are listed under
[Models](README.md).

### Purpose

`gaussian_process` is OPAL's uncertainty-aware regressor. It predicts:

- objective inputs (`y_pred`), and
- predictive standard deviation (`y_pred_std`) used by uncertainty-aware objectives/selection.

### Runtime contract

- Input: `X` as `float` matrix `(N, F)`.
- Output mean: always 2D `(N, D)` in OPAL runtime.
- Output standard deviation:
  - emitted in RoundCtx at `model/<self>/std_devs`,
  - reshaped to match scalar-output `(N, 1)` cases,
  - validated finite and non-negative (zeros allowed).

If `std=False` is requested by caller paths, the plugin returns only mean predictions and does not emit std chunks.

### Core math

Given training data `(X, Y)`, GP regression defines:

- posterior mean `mu(x)`,
- posterior standard deviation `sigma(x)`.

OPAL consumes `mu(x)` as model predictions and `sigma(x)` as uncertainty carrier (not variance).

### Config surface

`model.name: gaussian_process` with strict typed params in `campaign.yaml`:

- `alpha`, `normalize_y`, `random_state`, `n_restarts_optimizer`, `optimizer`, `copy_X_train`
- `kernel` block:
  - `name: rbf|matern|rational_quadratic|dot_product`
  - kernel-specific shape/scale params and bounds
  - optional white-noise settings

See [Configuration](../../reference/configuration.md) for full schema.

### Units invariant with `training.y_ops`

When `training.y_ops` is configured, OPAL inverse-transforms:

- `y_pred` mean, and
- `y_pred_std`

before objective evaluation. This keeps score channels and uncertainty channels in the same objective units.

If a model emits standard deviation and a configured y-op lacks `inverse_std`, OPAL fails fast.

### Error cases

- non-finite or negative predicted standard deviation
- shape mismatch between `y_pred` and `y_pred_std` in runtime path
- invalid kernel/config shape resolved by strict schema/loader checks

### Use in a campaign

The model participates in the same [campaign round](../../workflows/campaign-round.md)
as every other model. `top_n` needs a score channel. `expected_improvement`
also needs a standard-deviation channel emitted by the configured objective.

To confirm uncertainty is present in a completed round:

```bash
uv run opal ctx audit -c configs/campaign.yaml --round latest
```
