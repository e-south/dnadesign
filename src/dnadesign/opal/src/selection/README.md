# OPAL Selection Strategies

Selection strategies convert **scores** into **ranks and selected flags**.

## Contract

```python
def selection_fn(ids: np.ndarray, scores: np.ndarray, *, top_k: int, tie_handling: str = "competition_rank") -> dict
```

#### Inputs

- `ids` (N,): design identifiers
- `scores` (N,): scalar selection scores
- `top_k`: nominal selection count
- `tie_handling`: "competition_rank" (include all ties at boundary)

#### Outputs

- dict with fields:
    - `order_idx`: np.ndarray of indices sorted by (-score, id)
    - `rank_competition`: np.ndarray of ranks (1,2,3,3,5)
    - `selected_bool`: np.ndarray of booleans (include all ties)