## OPAL Models — Registry

### Inventory

| name | X expected | Y expected | key behavior |
| --- | --- | --- | --- |
| `random_forest` | `X: (N, F)` | `Y: (N,)` or `(N,D)` | ensemble regressor with OOB diagnostics |
| `gaussian_process` | `X: (N, F)` | `Y: (N,)` or `(N,D)` | GP regression with predictive std emitted to RoundCtx |

Quick check:

```python
from dnadesign.opal.src.registries.models import list_models
print(list_models())
```

### Contract

A model plugin must implement:

```python
class Model:
    def fit(self, X: np.ndarray, Y: np.ndarray, *, ctx: PluginCtx | None = None) -> Any: ...
    def predict(self, X: np.ndarray, *, ctx: PluginCtx | None = None) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...
    def load(path: str | Path) -> "Model": ...
```

### Gaussian Process params (v2)

`model.name: gaussian_process` accepts strict typed params:

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

### Runtime uncertainty carrier

`gaussian_process` emits predictive standard deviation into:

- `model/<self>/std_devs`

Runtime forwards this to objectives as `y_pred_std`. Objectives can emit named uncertainty channels; selection consumes uncertainty only through explicit `uncertainty_ref`.
For `expected_improvement`, that referenced uncertainty channel must be a standard deviation.

When `training.y_ops` are configured, OPAL inverse-transforms both:

- predictive means (`y_pred`), and
- predictive standard deviations (`y_pred_std`)

before objective evaluation, so `score_ref` and `uncertainty_ref` are always in the same objective units.
If a model emits standard deviations and any configured y-op does not implement `inverse_std`, the run fails fast.

### Extending

1. Implement a model wrapper satisfying the contract.
2. Register it with `@register_model("name")`.
3. Import the module in `src/dnadesign/opal/src/models/__init__.py`.
