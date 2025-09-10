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
| `random_forest` | `X: (N, F)` | `Y: (N,)` or `(N,D)` | uncertainty: per‑tree std · y‑scaling: robust IQR · OOB |

Quick check:

```python
from dnadesign.opal.registries.models import list_models
print(list_models())  # e.g., ["random_forest", ...]
```

---

## Contracts

The **registry** maps a model name → factory. A factory must return an object that implements **at minimum**:

```python
class Model:
    def fit(self, X: np.ndarray, Y: np.ndarray) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def save(self, path: str | Path) -> None: ...
    def load(path: str | Path) -> "Model": ...
```

**Registry helpers**

* `register_model(name: str)` — decorator to register a factory under `name`
* `get_model(name, *factory_args, **factory_kwargs)` — build by name
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
from dnadesign.opal.registries.models import register_model

@register_model("my_model")
def _factory(params: dict, *args, **kwargs):
    return MyModel(**params)
```
```python
# src/dnadesign/opal/src/models/__init__.py
from . import my_model  # noqa: F401
```
```yaml
models:
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

* **Per‑target robust scaling (IQR):** during **fit** only, each target is median‑centered and scaled by **IQR/1.349**; predictions are inverse‑transformed to original units. Guards: skip when labeled `N` is small (configurable) or scale≈0.
* **Uncertainty:** `predict_per_tree` returns per‑tree predictions; compute `std_vec = per_tree.std(axis=0, ddof=1)` (original units).
* **OOB diagnostics:** with `bootstrap: true` and `oob_score: true`, returns `FitMetrics(oob_r2, oob_mse)` (MSE in scaled space).
* **Persistence:** joblib bundle includes sklearn model and scaler state.

**Notes**

* Scaling is a training‑time aid only; all reported predictions are in original units.
* `std_vec` is tree dispersion (not calibrated probability).
