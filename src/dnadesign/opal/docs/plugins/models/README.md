<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[OPAL docs](../../index.md) / [Prediction and selection](../README.md)

# Prediction models

Models learn the relationship between candidate features (X) and measured
outcomes (Y), then predict Y for the remaining candidates. The objective scores
those predictions; the selection rule chooses the next batch.

## Inventory

| name | X expected | Y expected | key behavior |
| --- | --- | --- | --- |
| `random_forest` | `X: (N, F)` | `Y: (N,)` or `(N,D)` | ensemble regressor with OOB diagnostics |
| `gaussian_process` | `X: (N, F)` | `Y: (N,)` or `(N,D)` | GP regression with predictive std emitted to RoundCtx |

`expected_improvement` is not GP-only, but it requires an objective uncertainty channel referenced by `uncertainty_ref`.

Choose the model in `campaign.yaml`. For the random-forest example:

```yaml
# Fit 64 trees with a fixed seed for reproducible example runs.
model:
  name: random_forest
  params:
    n_estimators: 64
    random_state: 7
```

The [random-forest demo](../../../campaigns/demo_rf_topn/README.md) provides a
complete configuration. Use the [Gaussian-process demo](../../../campaigns/demo_gp_ei/README.md)
to try a selection rule that also uses predictive uncertainty.

## Model detail pages

- [Gaussian Process behavior and math](gaussian-process.md)

## Implement a model

A model plugin must implement the following interface. This is a signature
reference for plugin authors, rather than a standalone program:

```python
class Model:
    def fit(self, X: np.ndarray, Y: np.ndarray, *, ctx: PluginCtx | None = None) -> Any: ...
    def predict(self, X: np.ndarray, *, ctx: PluginCtx | None = None) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path, params: dict | None = None) -> "Model": ...
```

## Gaussian Process params (v2)

`model.name: gaussian_process` uses explicit typed params:

- `alpha`: positive scalar or list of positive values
- `normalize_y`: bool
- `random_state`: int or null
- `n_restarts_optimizer`: int >= 0
- `optimizer`: non-empty string or null
- `copy_X_train`: bool
- `kernel`: optional typed block

Kernel block:

- `name`: `rbf|matern|rational_quadratic|dot_product`
- `length_scale`, `length_scale_bounds.lower`, `length_scale_bounds.upper`
- `nu` (matern)
- `alpha`, `alpha_bounds.*` (rational_quadratic)
- `sigma_0`, `sigma_0_bounds.*` (dot_product)
- `with_white_noise`, `noise_level`, `noise_level_bounds.*`

## Runtime uncertainty carrier

`gaussian_process` emits predictive standard deviation into:

- `model/<self>/std_devs`

Runtime forwards this to objectives as `y_pred_std`. Objectives can emit named uncertainty channels; selection consumes uncertainty only through explicit `uncertainty_ref`. For `expected_improvement`, that referenced uncertainty channel must be a standard deviation.

When `training.y_ops` are configured, OPAL inverse-transforms both:

- predictive means (`y_pred`), and
- predictive standard deviations (`y_pred_std`)

before objective evaluation, so `score_ref` and `uncertainty_ref` are always in the same objective units. If a model emits standard deviations and any configured y-op does not implement `inverse_std`, the run stops with an error.

## Extending

1. Implement a model wrapper satisfying the contract.
2. Register it with `@register_model("name")`.
3. Import the module in `src/dnadesign/opal/src/models/__init__.py`.

## See also

- [Selection plugins](../selection/README.md)
- [Pool-relative weighted acquisition](../selection/expected-improvement.md)
- [Objective plugins](../objectives/README.md)
- [Campaign round](../../workflows/campaign-round.md)
