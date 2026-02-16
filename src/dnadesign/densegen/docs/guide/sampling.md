## Sampling (Stage-A + Stage-B)

This page explains how DenseGen chooses sites before solving.

Sampling happens in two steps:

- **Stage-A (inputs -> pools)**: mine or ingest candidate TFBS rows and retain `n_sites`
- **Stage-B (pools -> libraries)**: build smaller solver libraries from Stage-A pools

Quick map:

```
densegen.inputs[].sampling   (Stage-A) -> outputs/pools/<input>__pool.parquet
densegen.generation.sampling (Stage-B) -> outputs/libraries/* + outputs/tables/attempts.parquet
```

Use this guide for behavior and intent. Use [../reference/config.md](../reference/config.md)
for exact field names and allowed values.

Two knobs are easy to confuse:

- `mining.budget.candidates`: how many candidates Stage-A evaluates (search effort)
- `sampling.n_sites`: how many rows Stage-A keeps in the final pool (retained size)

If you want better diversity/coverage without exploding solver input size, increase
`candidates` first, then tune `n_sites`.

---

### Stage-A for PWM inputs

Stage-A for PWM inputs is a mine -> score -> dedupe -> retain loop:

1) **Generate cores** (and optional flanks) via `sampling.strategy`.
2) **Score** with FIMO log-odds (`--norc` forward only). If `sampling.bgfile` is set, that background is also used for theoretical max, `score_norm`, and MMR information-content weights.
3) **Eligibility**: candidate must have a FIMO hit and `best_hit_score > 0`.
4) **Deduplication** via `uniqueness.key`:
   - `sequence` keeps unique full TFBS strings.
   - `core` collapses by `tfbs_core` (motif-aligned match).
   - For multi-motif PWM inputs, `uniqueness.cross_regulator_core_collisions`
     controls what happens when the same `tfbs_core` appears under different regulators
     (`warn` by default, `error` for fail-fast, `allow` to disable).
5) **Selection**: retain `n_sites` using `selection.policy`.

Important: Stage-A PWM sampling is **not** full randomness. It is PWM-biased sampling,
and that happens before FIMO. Only the target length and background padding are random.
The core itself is sampled from the PWM’s per-position probabilities, so highly specific
motifs will repeatedly generate the same cores.

What actually happens for PWM inputs when `sampling.strategy=stochastic`:

1) Sample a target length (e.g., 16–20).
2) Sample a core by drawing each position from the PWM (or the trimmed PWM window).
3) Embed the core in random background bases to reach the target length.
4) Run FIMO to score and get `matched_sequence` (core).
5) Collapse by `tfbs_core` → `eligible_unique` count.

#### Length range, padding, and trimming

When `sampling.length.policy=range`, Stage‑A samples a **target length per candidate**
uniformly from `[min, max]`. This happens per generated sequence.

- **Short motifs (width < target length)**: the motif is embedded in background bases.
  The left pad length is sampled uniformly from `0..extra`, and the right pad receives
  the remaining bases. Each candidate gets its own left/right split.
- **Long motifs (width > target length)**: Stage‑A trims to the **max‑information contiguous
  window** of the target length, then runs FIMO on that trimmed motif window.

Selection uses `score_norm = best_hit_score / pwm_theoretical_max_score`, where the
theoretical max is computed from the **same window length** used for the candidate:

- if `target length >= motif width`, the full motif max is used
- if `target length < motif width`, the max-info window of that target length is used

This keeps `score_norm` comparable across mixed target lengths and prevents longer
targets from being favored solely because they allow higher raw scores.

#### Selection policies

- `top_score`: take the top `n_sites` by `selection.rank_by` (default `best_hit_score`).
- `mmr`: Maximal Marginal Relevance (Carbonell & Goldstein).
  Ranking/tiering/pool order is controlled by `selection.rank_by` (`score` or `score_norm`).

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

MMR requires **uniform core length**. If you set `sampling.length.policy=range` with
a minimum below the motif width, configure `sampling.trimming.window_length` to a
fixed length (or keep the length range >= motif width) so all cores are the same length.
When a fixed trimming window is set, Stage‑A logs the configured window and records
per‑motif trimming metadata in `outputs/pools/pool_manifest.json` (stage‑A sampling
histogram entries include `motif_width`, `trimmed_width`, `trim_window_length`,
`trim_window_strategy`, `trim_window_start`, `trim_window_score`, and `trim_window_applied`).

`selection.pool.min_score_norm` is an active MMR pool gate using the **fraction of theoretical
max log-odds score**:
`best_hit_score / pwm_theoretical_max_score`. Candidates below the threshold are excluded before
MMR selection. There is no default.

#### Stage-A output contract

Stage-A writes a single pool parquet per input under `outputs/pools/<input>__pool.parquet`.
This parquet contains **only the retained set** (the output of the configured selection policy).
When `selection.policy=mmr`, this is the diversified set. Plots may compare against a top-score
baseline for diagnostics, but that baseline is not written as a separate pool.

#### Candidate logging + length inspection

To keep all candidates for debugging, set:

```
densegen.inputs[].sampling.keep_all_candidates_debug: true
```

Then after `dense stage-a build-pool`, inspect candidate length distributions from
the per-input parquet under `outputs/pools/candidates/<input_name>/`. Compare:

- **All candidates**: length distribution across `[min, max]`
- **Eligible candidates** (`best_hit_score > 0`): length distribution after FIMO

Use your preferred analysis environment to compute length counts from the
`candidates__<motif_id>.parquet` file for the motif you care about.

#### Tiers + MMR pool

`sampling.tier_fractions` defines diagnostic tiers and the **rung ladder** for MMR frontier
selection. Rungs are evaluated by **post-gate count** (after `selection.pool.min_score_norm`).
DenseGen then slices that frontier to a deterministic MMR target pool (`ceil(10.0 * n_sites)`,
bounded by available candidates and `selection.pool.max_candidates` when set). This avoids large,
stepwise pool jumps at rung boundaries.
In `outputs/plots/stage_a/pool_tiers.pdf`, dashed lollipops show the configured diagnostic tier
boundaries, and one solid lollipop marks the worst retained percentile after MMR across
non-background regulators.

#### Mining budget

`mining.budget.mode` controls how long Stage-A mines:

- `fixed_candidates` (recommended): score exactly `budget.candidates`, then retain.
- `tier_target` (advanced): mine until enough eligible uniques exist to fill a target tier
  or until caps/time are reached.

When tier targeting is unmet, DenseGen retains the best available set and records the shortfall
in `outputs/pools/pool_manifest.json`.

---

### Stage-B sampling

Stage-B builds solver libraries from Stage-A pools. Stage-B always consumes the Stage-A pool parquet **as-written** (as-selected).

Stage‑B sampling builds/rebuilds libraries during `dense run` as runtime resamples
occur and records them under `outputs/libraries/`.

Key control is `densegen.generation.sampling.pool_strategy`:

- `full`: use the entire pool (good for tiny pools).
- `subsample`: build libraries of `library_size` from the pool and offer them to the solver.
- `iterative_subsample`: repeatedly rebuild libraries within runtime caps (best for large pools).

When `pool_strategy: full`, avoid setting `library_size`, `library_sampling_strategy`,
`cover_all_regulators`, `max_sites_per_regulator`, or `relax_on_exhaustion`; they do not apply.

Additional Stage-B controls that commonly affect "it fails vs it works":

- **Uniqueness + exhaustion**
  - `unique_binding_sites` / `unique_binding_cores` prevent duplicate sites in a library.
  - `cover_all_regulators` attempts to include every regulator in the library at least once (defaults to false).
  - If the pool cannot satisfy these constraints at `library_size`, Stage‑B errors.
  - `relax_on_exhaustion` only relaxes per‑TF caps (`max_sites_per_regulator`).
- **Sampling strategies**
  - `tf_balanced`: balances sampling across regulators.
  - `uniform_over_pairs`: encourages pair coverage (useful when co-occurrence matters).
  - `coverage_weighted`: boosts under-covered regulators/sites; optional failure penalties can
    down-weight motifs that frequently fail solves.

Operational note: resampling is driven by runtime controls such as stalls/duplicate pressure
and `runtime.max_consecutive_failures` (see `densegen.runtime.*`). When diagnosing a run, `placement_map` and
`dense inspect run --events --library` are the fastest way to see whether the run is rebuilding
libraries too often or sampling is exhausted.

Stage‑B warns when the total bp in a sampled library is below `generation.sequence_length`.
Solver coverage may be limited in that case; increase `library_size` or supply longer motifs if you
need full coverage.

For plan constraints and solver configuration, see `guide/generation.md`.

---

@e-south
