## Sampling (Stage-A + Stage-B)

DenseGen sampling is staged:

- **Stage-A (inputs -> pools)**: mine or ingest TFBS candidates per input and retain `n_sites`.
- **Stage-B (pools -> libraries)**: repeatedly build small solver libraries from those pools.

Quick map:

```
densegen.inputs[].sampling   (Stage-A) -> outputs/pools/<input>__pool.parquet
densegen.generation.sampling (Stage-B) -> outputs/libraries/* + outputs/tables/attempts.parquet
```

This guide is semantic. For exact schema fields, use `reference/config.md`.

---

### Stage-A for PWM inputs

Stage-A for PWM inputs is a mine -> score -> dedupe -> retain loop:

1) **Generate cores** (and optional flanks) via `sampling.strategy`.
2) **Score** with FIMO log-odds (`--norc` forward only). Cores are treated as bricks and can be
   placed in either orientation later.
3) **Eligibility**: candidate must have a FIMO hit and `best_hit_score > 0`.
4) **Deduplication** via `uniqueness.key`:
   - `sequence` keeps unique full TFBS strings.
   - `core` collapses by `tfbs_core` (motif-aligned match).
5) **Selection**: retain `n_sites` using `selection.policy`.

#### Selection policies

- `top_score`: take the top `n_sites` by `best_hit_score`.
- `mmr`: Maximal Marginal Relevance (Carbonell & Goldstein).

MMR definition used:

```
utility(x) = alpha * relevance(x) - (1 - alpha) * max_similarity_to_selected
```

Relevance:
- `selection.pool.relevance_norm = minmax_raw_score` (default): `score_norm = best_hit_score / pwm_theoretical_max_score`
- `percentile`: percentile rank within the MMR pool

Similarity:
- derived from weighted Hamming on `tfbs_core` with
  `similarity = 1 / (1 + dist)`.

`selection.pool.min_score_norm` is a **report-only** reference for "within tau of theoretical max."
It does not filter the MMR pool. There is no default; set it explicitly if you want the reference.

#### Tiers + MMR pool

`sampling.tier_fractions` defines diagnostic tiers and the **rung ladder** for MMR pool selection.
The pool is the smallest rung that can supply `n_sites` (or the full list if none can).
An explicit `selection.pool.max_candidates` cap truncates the pool by score if set.

#### Mining budget

`mining.budget.mode` controls how long Stage-A mines:

- `fixed_candidates` (recommended): score exactly `budget.candidates`, then retain.
- `tier_target` (advanced): mine until enough eligible uniques exist to fill a target tier
  or until caps/time are reached.

When tier targeting is unmet, DenseGen retains the best available set and records the shortfall
in `outputs/pools/pool_manifest.json`.

---

### Stage-B sampling (brief)

Stage-B builds solver libraries from Stage-A pools. It is the **only** stage that resamples
during a run. Key control is `densegen.generation.sampling.pool_strategy`:

- `full`: use the entire pool (good for tiny pools).
- `subsample`: build new libraries of `library_size` per attempt.
- `iterative_subsample`: resample libraries repeatedly within runtime caps.

For plan constraints and solver configuration, see `guide/generation.md`.

---

@e-south
