# OPAL Architecture and Data Flow

- Models are not aware of downstream objectives (see [RoundCtx](./roundctx.md)).
- Objectives derive their own round constants via `train_view` and publish them.
- Selection can read whatever objectives produced.
- Persisted `round_ctx.json` makes runs auditable alongside ledger sinks, `model.joblib`, `model_meta.json`,
  `selection_top_k.csv`, and `objective_meta.json`.

```bash
# Labels
my_experimental_data.csv -> transforms_y -> labels [id, y(list<float>)] -> appends event: label

# Features
records.parquet [X] -> transforms_x -> X (fixed width)

# Train & score
X + labels -> model.fit -> predict Y_hat -> objective -> pred__y_obj_scalar -> selection (top-k)

# Canonical ledger sinks
outputs/ledger.*: { label | run_pred | run_meta }
```
