# OPAL Models — Registry

### Contents

1. [Inventory](#inventory)
2. [Contracts](#contracts)
3. [Wiring](#wiring)
4. [Extending](#extending)
5. [Entries](#entries)
    i. [random\_forest](#random_forest)

## Inventory

Dashboard of current entries (see details below):

| name            | X expected  | Y expected           | features (brief)                                        |
| --------------- | ----------- | -------------------- | ------------------------------------------------------- |
| `random_forest` | `X: (N, F)` | `Y: (N,)` or `(N,D)` | uncertainty: per-tree std · OOB |

Quick check:

```python
from dnadesign.opal.src.registries.models import list_models
print(list_models())  # e.g., ["random_forest", ...]
```

---

## Contracts

The **registry** maps a model name → factory. A factory must return an object that implements **at minimum**:

```python
class Model:
    def fit(self, X: np.ndarray, Y: np.ndarray, *, ctx: PluginCtx | None = None) -> Any: ...
    def predict(self, X: np.ndarray, *, ctx: PluginCtx | None = None) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...
    def load(path: str | Path) -> "Model": ...
```

**Registry helpers**

* `register_model(name: str)` — decorator to register a factory under `name`
* `get_model(name, params: dict)` — build by name (factory must accept a params dict)
* `list_models() -> list[str]` — enumerate available entries

**Paths**
Registry: `src/dnadesign/opal/src/registries/models.py`
Implementations: `src/dnadesign/opal/src/models/`

---

## Extending

1. Implement a wrapper that satisfies the contract.
2. Register a factory and import it in `models/__init__.py`.

```python
# src/dnadesign/opal/src/models/my_model.py
from dnadesign.opal.src.registries.models import register_model

@register_model("my_model")
def _factory(params: dict, *args, **kwargs):
    return MyModel(**params)
```
```python
# src/dnadesign/opal/src/models/__init__.py
from . import my_model  # noqa: F401
```
```yaml
model:
  name: "<model_name>"
  params:
    # model‑specific parameters go here
```

---

## Entries

### random\_forest

**File:** `src/dnadesign/opal/src/models/random_forest.py`
**Backend:** `sklearn.ensemble.RandomForestRegressor`

**Implements (contract):** `fit`, `predict`, `save`, `load`
**Also provides:** `predict_per_tree(X)->(T,N,D)`, `feature_importances()`, `get_params()`

**Shapes**

* `X: (N, F)`
* `Y: (N,)` *or* `(N, D)` (multi‑output supported)

**Features**

* **Uncertainty:** `predict_per_tree` returns per-tree predictions; compute `std_vec = per_tree.std(axis=0, ddof=1)`.
* **OOB diagnostics:** with `bootstrap: true` and `oob_score: true`, returns `FitMetrics(oob_r2, oob_mse)`.
* **No target scaling inside the model:** apply any label scaling via **training `y_ops`** (e.g., `intensity_median_iqr`), which the runner fits/applies and inverts at prediction time.

**Notes**

* Scaling is a training‑time aid only; all reported predictions are in original units.
* `std_vec` is tree dispersion (not calibrated probability).
* `model_meta.json` now records `training__y_ops` so prediction can enforce Y‑ops inversion.

#### Runtime carrier contracts

Models may declare `@roundctx_contract(category="model", ...)` to enforce and audit
runtime keys in `round_ctx.json`. If a contract is declared, OPAL enforces it on
`fit` and `predict` when a `ctx` is provided.
