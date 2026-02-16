# OPAL Architecture and Data Flow

- Models are not aware of downstream objectives (see [RoundCtx](./roundctx.md)).
- Objectives derive their own round constants via `train_view` and emit named score/uncertainty channels.
- Selection reads only the channels referenced in config (`score_ref`, optional `uncertainty_ref`).
- Persisted `round_ctx.json` makes runs auditable alongside ledger sinks, `model.joblib`, `model_meta.json`,
  `selection_top_k.csv`, and `objective_meta.json`.

```bash
# Labels
my_experimental_data.csv -> transforms_y -> labels [id, y(list<float>)] -> appends event: label

# Features
records.parquet [X] -> transforms_x -> X (fixed width)

# Train & score
X + labels -> model.fit -> predict y_hat (+optional std) -> objectives -> channel refs -> selection (top-k)

# Canonical ledger sinks
outputs/ledger.*: { label | run_pred | run_meta }
```
