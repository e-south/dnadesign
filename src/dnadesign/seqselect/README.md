 ## seqelect -- Active Learning–Driven Candidate Prioritization
A submodule for **prioritized sequence selection** in low-N active learning regimes, leveraging model predictions, uncertainty, and orthogonal diversity metrics.

---

## 🔍 Motivation

In dense regulatory sequence design, testing throughput is often constrained. The goal is to iteratively select *informative and diverse batches of candidate sequences* for synthesis and labeling (e.g., 16 per round). These selections must balance:

- **Exploitation**: predicted functional output from the model  
- **Exploration**: uncertainty and functional novelty  
- **Diversity**: literal sequence orthogonality and regulatory program coverage

`smartselect` provides a scoring engine for candidate ranking using Evo2 latent embeddings, SW alignment metrics, model outputs, and optional clustering.

---

## 🧠 Summary of Selection Strategy

For each unlabeled candidate sequence, compute a **composite acquisition score** that integrates:

| Term | Description | Role |
|------|-------------|------|
| `y_pred` | Predicted function (from trained regressor) | Exploitation |
| `uncertainty` | Model variance / prediction disagreement | Ambiguity |
| `latent_dissimilarity` | Mean/max Evo2 distance to labeled set | Regulatory novelty |
| `literal_dissimilarity` | Mean/max SW distance to labeled set | Scaffold novelty |
| `cluster_redundancy` | Density of labeled points in candidate’s Evo2 cluster | Penalize resampling |
| *(optional)* `latent_density` | Local Evo2 point density | Outlier rejection |
| *(optional)* `prior_acq_density` | Reuse of candidate’s region in past rounds | Promote long-term spread |

All terms are normalized to [0, 1] per batch and combined via a weighted sum.

---

## 🛠️ Functional Overview

```python
def rank_candidates(
    X_latent: np.ndarray,
    SW_matrix: np.ndarray,
    labeled_indices: List[int],
    model: BaseEstimator,
    uncertainty_fn: Callable,
    cluster_ids: Optional[np.ndarray] = None,
    acquisition_history: Optional[List[int]] = None,
    config: Dict[str, float]  # α, β, γ, δ, ε...
) -> pd.DataFrame:
    """
    Returns a ranked DataFrame of candidate indices and their acquisition scores.
    """
```

Output:
```python
pd.DataFrame([
    {"index": i, "score": ..., "y_pred": ..., "uncertainty": ..., "latent_dissimilarity": ..., "sw_dissimilarity": ..., ...}
    for i in range(N_candidates)
])
```

---

## ⚙️ Configurable Weights (YAML compatible)

```yaml
smartselect:
  weights:
    y_pred: 0.4
    uncertainty: 0.2
    latent_dissimilarity: 0.2
    literal_dissimilarity: 0.1
    cluster_redundancy: 0.1
  diversity_metrics:
    latent: cosine
    literal: sw_mean
  use_optional:
    latent_density: false
    prior_acq_density: false
```

---

## 🔁 Integration with Active Loop

Typical usage within a round:

1. Train model on labeled sequences.
2. Predict `y_pred` and `uncertainty` for unlabeled pool.
3. Use `smartselect.rank_candidates(...)` to prioritize.
4. Select top K (e.g., 16) for next synthesis round.
5. Update labeled set and repeat.

---

## 🧪 Future Extensibility

- Plug in alternate diversity metrics (e.g., edit distance)
- Pareto-front selection mode instead of scalar ranking
- Support categorical outputs (classifiers) for labeling

---

## 🏷️ Naming Justification

- `smartselect` aligns with your other intuitive submodule names like `libshuffle` and `billboard`
- Emphasizes **selective intelligence** under constraint
- Can expose CLI subcommands like:
  ```bash
  python -m smartselect.rank --config configs/select.yaml
  ```

---

Let me know if you'd like a ready-to-implement scaffold including scoring, normalization, and prioritization logic inside a `smartselect/` directory.