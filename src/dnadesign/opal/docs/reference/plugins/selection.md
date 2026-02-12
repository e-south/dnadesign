## OPAL Selection Strategies

Selection strategies convert **scores** into **ranks and selected flags**.

### Contract

```python
def selection_fn(
    *,
    ids: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    objective: str = "maximize",
    tie_handling: str = "competition_rank",
    ctx: PluginCtx | None = None,
) -> dict
```

#### Inputs

- `ids` (N,): design identifiers
- `scores` (N,): scalar selection scores
- `top_k`: nominal selection count
- `objective`: "maximize" | "minimize"
- `tie_handling`: "competition_rank" (include all ties at boundary)
- `ctx`: RoundCtx plugin context (for contract enforcement/audit)

#### Outputs

- dict with fields:
    - `order_idx`: np.ndarray of indices sorted by (-score, id) (optional; OPAL will compute if absent)

OPAL normalizes to canonical arrays (`rank_competition`, `selected_bool`) after the plugin returns.

#### Runtime carrier contracts

Selection plugins may declare `@roundctx_contract(category="selection", ...)` to record
consumed/produced keys in `round_ctx.json`. If a contract is declared, OPAL enforces it.
