## RoundCtx and Contract Auditing

`RoundCtx` is a runtime companion to `campaign.yaml`: YAML chooses plugin graph. `RoundCtx`tracks what each stage computes and what downstream components consume. Use this page to understand runtime carriers, contract enforcement, and audit keys.

The runner persists:

- `round.log.jsonl`: stage events and batch progress
- `round_ctx.json`: full runtime carrier snapshot for audit

### Plugin-scoped usage

The runner injects a plugin-scoped `ctx` that expands `"<self>"` to the plugin name and enforces contracts.

```python
from dnadesign.opal.src.core.round_context import roundctx_contract

@roundctx_contract(
  category="objective",
  requires=["core/labels_as_of_round"],
  produces=["objective/<self>/foo"],
)
def my_objective_plugin(..., ctx=None, train_view=None): ...
```

Rules:

- `ctx.get(...)` may read any existing key.
- `ctx.set(...)` may only write keys declared by `produces` (or `produces_by_stage` when stage-scoped).
- Undeclared writes fail fast with a contract error.

Audit fields in `round_ctx.json`:

- `core/contracts/<category>/<plugin>/consumed`
- `core/contracts/<category>/<plugin>/produced`

### Batched predict stage behavior

For contracts using `produces_by_stage` (for example model `predict` summaries):

1. `ctx.precheck_requires(stage="predict")` activates stage checks.
2. Writes to keys declared in `produces_by_stage["predict"]` are staged in memory.
3. `ctx.get(...)` is read-your-writes for staged keys during the active stage.
4. `ctx.postcheck_produces(stage="predict")` commits staged keys once (last write wins in-stage).
5. Non-stage keys remain immediate and immutable (`allow_overwrite=False` behavior).

This preserves persisted RoundCtx immutability while allowing batch-by-batch accumulation.

### Predict-stage accumulation pattern

Use `ctx.get(...)` + `ctx.set(...)` on a stage-scoped key to update summaries across batches:

```python
prev = ctx.get("model/<self>/predict_summary", default={"n_batches": 0, "n_rows": 0})
next_summary = {
  "n_batches": prev["n_batches"] + 1,
  "n_rows": prev["n_rows"] + int(len(batch_df)),
}
ctx.set("model/<self>/predict_summary", next_summary)
```

Within the active predict stage, `ctx.get` reads staged values first. At stage end, OPAL commits the final staged value once.

### Validation lifecycle

1. Runner builds `RoundCtx` with core keys and plugin names.
2. Runner creates plugin-scoped contexts and checks `requires`.
3. Plugins execute using `ctx.get(...)` / `ctx.set(...)`.
4. Runner checks `produces` and writes `round_ctx.json`.

Inspect carriers via CLI:

- `opal ctx show`
- `opal ctx audit`
- `opal ctx diff`
