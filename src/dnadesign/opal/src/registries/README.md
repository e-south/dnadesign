# OPAL Registries

Plugin points that keep OPAL decoupled. Each plugin is discovered via a simple
`register_*` decorator and looked up by name from YAML.

Layout:
```
src/registries/
  __init__.py      # convenience re-exports
  transforms_y.py  # CSV → labels (id,y)
  transforms_x.py  # X-column → matrix
  objectives.py    # Ŷ → score (+diagnostics)
  selections.py    # scores → ranks/selected
  models.py        # model factory
```

Contracts (minimal, stable):

- transforms_x
  - register: `@register_rep_transform("name")`
  - factory signature: `(params: dict) -> transformer`
  - transformer API: `.transform(series: pd.Series) -> (X: np.ndarray, x_dim: int)`

- transforms_y
  - register: `@register_ingest_transform("name")`
  - function signature: `(df_tidy: pd.DataFrame, params: dict, setpoint: list[float]) -> (df_id_y: pd.DataFrame, meta: dict)`
  - return `df_id_y` with columns `['id','y']` where `y` is list[float] with a shape negotiated by the campaign

- models
  - register: `@register_model("name")`
  - factory signature: `(params: dict, target_scaler_cfg: dict) -> model`
  - model API: `.fit(X,Y) -> metrics`, `.predict(X) -> Yhat`, `.predict_per_tree(X) -> np.ndarray` (optional), `.save(path)`, `.load(path)`

- objectives
  - register: `@register_objective("name")`
  - function signature: `(*, y_pred: np.ndarray, params: dict, round_ctx: RoundContext) -> ObjectiveResult`
  - returns `ObjectiveResult(score: np.ndarray, diagnostics: dict)`

- selections
  - register: `@register_selection("name")`
  - function signature: `(ids: np.ndarray, scores: np.ndarray, *, params: dict) -> { order_idx, ranks, selected_bool }`

See the main README for how these plug into the round flow. Keep plugins
stateless and deterministic where possible.
