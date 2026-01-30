## Sampling (Stage‑A + Stage‑B)

DenseGen separates “sampling” into two different steps, with different purposes and artifacts:

- **Stage‑A (inputs → pools)**: realizes each configured input into a concrete pool of TF binding sites.
- **Stage‑B (pools → libraries)**: repeatedly samples **solver libraries** (small subsets) from those pools according to a policy (coverage, uniqueness, weighting) and offers each library to the optimizer.

A practical mental model:

```
densegen.inputs[].sampling    (Stage‑A)  → outputs/pools/<input>__pool.parquet
densegen.generation.sampling  (Stage‑B)  → outputs/libraries/*  + outputs/tables/attempts.parquet
solver (dense-arrays)                    → outputs/tables/dense_arrays.parquet
```

This page describes **what Stage‑A** and **Stage‑B** do (semantics and invariants). For exact field names and schema shape, see the config reference.

---

### Stage‑A sampling

Stage‑A configuration lives under `densegen.inputs[].sampling`. You run it via:

- `dense stage-a build-pool` (recommended explicit step), or
- `dense run --rebuild-stage-a` (when you want the run itself to rebuild pools).

Stage‑A writes **cached pool artifacts** and does **not** resample during optimization.

---

### Stage‑A behavior by input type

Stage‑A always produces a pool, but “sampling” means different things depending on input type:

- **PWM-backed inputs** (`pwm_*`, `pwm_artifact*`): Stage‑A *generates candidates*, scores them with FIMO, deduplicates, tiers, and retains `n_sites` per regulator (details below).
- **Binding-site tables** (`type: binding_sites`): Stage‑A is ingestion + normalization; it does not mine. The pool is the table rows (with `tfbs_core` derived as the full TFBS sequence for core-uniqueness support).
- **Sequence libraries** (`type: sequence_library`, `type: usr_sequences`): Stage‑A is ingestion; the pool is a list of sequences (used as solver seeds depending on your workflow).

---

### Stage‑A for PWM inputs

PWM-backed Stage‑A is a “mine → score → dedupe → retain” loop per motif/regulator.

#### 1) Generate candidate sequences

Stage‑A generates a motif-width **core**, then (optionally) adds flanks.

- `strategy: stochastic`
  Sample a core from the PWM probability matrix.
- `strategy: consensus`
  Emit the single consensus site (requires `n_sites: 1`).
- `strategy: background`
  Sample the core from the PWM background distribution (useful as a negative control).

Optional preprocessing:
- `trimming.window_length` (with `trimming.window_strategy: max_info`) can trim a PWM to its
  highest-information window before sampling.

#### 2) Normalize length (optional flanks)

Length normalization is how DenseGen generates TFBS sequences that meet length requirements.

- `length.policy: exact`
  TFBS length equals motif width (no flanks).
- `length.policy: range` with `length.range: [min, max]`
For **each candidate**, choose a target length in `[min, max]`, then embed the core inside left/right flanks sampled from the PWM background. The core offset within the TFBS is randomized.

#### 3) Score with FIMO (log‑odds)

DenseGen invokes FIMO in score-first mode and records:

- `best_hit_score` (the best FIMO log‑odds score per candidate TFBS)

Key scoring semantics:

- FIMO is run with `--thresh 1.0` so the reporting threshold does not gate results.
- Stage‑A uses `--norc` (forward strand only) for scoring.

#### 4) Eligibility (hard gate)

A candidate is eligible **iff**:

- it has at least one FIMO hit, **and**
- `best_hit_score > 0`

This means Stage‑A is strictly score-based: no p-values, no q-values.

#### 5) Uniqueness (what counts as “the same site”)

Stage‑A deduplicates **eligible** candidates before ranking and retention. Controlled by `uniqueness.key`:

- `sequence`
  Unique by the full TFBS string (`tfbs`). Flanks matter.
- `core`
  Collapse candidates by a motif-aligned **core identity** derived from FIMO’s `matched_sequence`
  (`tfbs_core`). Flanks do not matter.

When multiple candidates map to the same core, Stage‑A keeps **the best representative**:
highest score, with deterministic tie-breakers.

Terminology used in outputs:

- `candidates_with_hit` — candidates with at least one FIMO hit (pre‑eligibility).
- `eligible_raw` — candidates that passed eligibility (`best_hit_score > 0`) before deduplication.
- `eligible_unique` — the deduplicated candidate set used for ranking/retention.

Pool columns you’ll commonly use:

- `tfbs` — full TFBS sequence (core + flanks)
- `tfbs_core` — motif-aligned core (from FIMO matched sequence; used for “core uniqueness”)
- `best_hit_score` — best log‑odds score
- `fimo_start`, `fimo_stop`, `fimo_strand` — best-hit metadata within the TFBS sequence

#### 6) Ranking + tiers (diagnostics, and budget targeting)

Eligible unique candidates are ranked **per regulator** by:

1) descending `best_hit_score`, then
2) lexicographic TFBS sequence (deterministic tie-break)

DenseGen assigns score tiers by rank fraction:

- tier 0: top 0.1%
- tier 1: next 1%
- tier 2: next 9%
- tier 3: remainder

Tiers are primarily for:
- diagnostics (`stage_a_summary`), and
- tier-target mining (next section)

Tier boundary scores are recorded in `outputs/pools/pool_manifest.json`.

#### 7) Retention: choose `n_sites` per regulator

After ranking (and deduplication), Stage‑A retains `n_sites` per regulator using `selection.policy`:

- `top_score` (default)
  Take the top `n_sites` by `best_hit_score` (deterministic).
- `mmr`
  Use Maximal Marginal Relevance to trade off score vs diversity while staying score-first.

MMR (high-level, faithful to implementation; after Carbonell & Goldstein, 1998):

- Utility:
  `utility = alpha * normalized_score - (1 - alpha) * max_similarity_to_selected`
- `alpha ∈ (0, 1]` biases toward score (`→ 1`) vs diversity (`→ 0`).
- `normalized_score` is the percentile rank within the candidate pool (0–1, ties averaged).
- Reporting uses `score_norm = best_hit_score / pwm_max_score` (PWM consensus log‑odds)
  for cross‑TF comparability; this is separate from the MMR percentile normalization used in selection.
- Similarity is derived from a **PWM‑tolerant weighted Hamming distance** on `tfbs_core`:

  - Information content per position:
    `IC_i = 2 - H_i` where `H_i = -Σ p_i(b) log2 p_i(b)` (bits).
  - PWM‑tolerant weights:
    `w_i = 1 - (IC_i / 2)` (so low‑information positions get higher weight).
  - Distance:
    `dist(x, y) = Σ w_i * [x_i ≠ y_i]`
- Similarity:
  `similarity = 1 / (1 + dist)`

PWM‑tolerant diversity (safer for PWM‑likeness):
- weights ~ (1 - IC)
- encourages diversity in low‑information (tolerant) positions
- preserves high‑information (specific) positions

Reference: Carbonell & Goldstein, “The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries.” (SIGIR 1998)
https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

Performance/behavior knobs for MMR:

- `selection.shortlist_*` controls how many of the top-ranked candidates are considered.
- `selection.tier_widening` can specify a ladder of ranked fractions to search (e.g. `[0.001, 0.01, 0.09, 1.0]`).
  DenseGen tries the first rung (top slice); if it can’t fill from that slice, it widens to the next rung.
- When tier widening is enabled, DenseGen widens until the candidate pool reaches
  `shortlist_target = max(shortlist_min, shortlist_factor * n_sites)` (or the ladder exhausts).
- When `selection.policy: mmr` and `selection.tier_widening` is omitted, DenseGen enables tier widening
  with the default ladder `[0.001, 0.01, 0.09, 1.0]`.

> Practical advice: MMR is best used when you expect **many** eligible unique candidates and you want
> to avoid near-duplicates while still staying near the score frontier.

#### 8) Mining budget: how many candidates to score

`mining.budget` controls how long Stage‑A mines candidates before retention:

- `mode: fixed_candidates`
  Score exactly `budget.candidates` candidates, then retain.
- `mode: tier_target`
  Mine until you have enough eligible uniques to plausibly fill `n_sites` from within a target rank fraction:

  `required_unique = ceil(n_sites / target_tier_fraction)`

  Mining grows the candidate target by `growth_factor` until either:
  - the tier target is met, or
  - `max_candidates` is reached, or
  - `max_seconds` is reached.

When tier targeting is unmet, DenseGen still retains the best available set and records the shortfall
in `outputs/pools/pool_manifest.json` so downstream audits stay honest.

---

### Stage‑B sampling

Stage‑B configuration lives under `densegen.generation.sampling`.

Conceptually: Stage‑B builds libraries (small TFBS subsets) from Stage‑A pools (or replays library artifacts),
then offers each library to the solver. **This is the only stage that resamples during a run.**

Stage‑B produces:

- canonical library artifacts under `outputs/libraries/`
- an attempt audit log under `outputs/tables/attempts.parquet`
- run summaries + events under `outputs/meta/*`

#### 1) Pick a pool strategy (how libraries are constructed)

- `pool_strategy: full`
  Library = the full pool.
  Good for tiny pools; reduces Stage‑B’s ability to escape weak mixes via resampling.

- `pool_strategy: subsample`
  Sample a new library of `library_size` TFBS from the pool (per input + plan).
  Good default when pools are large.

- `pool_strategy: iterative_subsample`
  Like `subsample`, but can rebuild libraries repeatedly during a run (bounded by runtime caps).
  Use when the solver stalls or yields too few new solutions per library.

> Note: Stage‑B can also replay prebuilt libraries (`library_source: artifact`) for deterministic re-runs.
> The semantics of “what a library contains” are the same; only the source differs.

#### 2) Enforce coverage + uniqueness inside the library (pre-solver)

These constraints apply **during library construction**, before the solver runs:

- `cover_all_regulators: true`
  Ensure each regulator appears at least once in the library. If a regulator has 0 sites in Stage‑A pools,
  Stage‑B fails fast (coverage is impossible).

- `unique_binding_sites: true`
  Deduplicate identical `(TF, TFBS)` pairs within a library.

- `unique_binding_cores: true`
  Deduplicate by `(TF, core)` within a library to avoid “same site, different flanks”.
  - For PWM inputs, `tfbs_core` comes from Stage‑A’s FIMO matched sequence.
  - For binding-site tables, `tfbs_core` defaults to the full TFBS sequence.

- `max_sites_per_regulator` (optional)
  Cap per-TF dominance when pools are large and uneven.

#### 3) Choose how the library is sampled (which sites are preferred)

`library_sampling_strategy` biases which sites are preferred when sampling libraries:

- `tf_balanced` — bias toward even TF representation.
- `uniform_over_pairs` — bias toward diversity of TF pairs.
- `coverage_weighted` — dynamically boosts under-used motifs based on run history so coverage improves over time.

For `coverage_weighted`, the boost shape is controlled by:

- `coverage_boost_alpha` (strength)
- `coverage_boost_power` (how sharply the boost grows)

#### 4) Offer each library to the solver

For each sampled library, the solver selects a subset of sites and places them into sequences subject to:

- `generation.sequence_length`
- plan constraints (`required_regulators`, `min_required_regulators`, per-TF minimums)
- fixed elements (e.g., promoter constraints)

Stage‑B records feasibility signals (bp accounting: fixed bp, required bp, slack) so you can distinguish:
- “sampling gave me weak libraries” vs
- “constraints physically don’t fit in the requested sequence length”.

---

### Artifacts and where truth is recorded

If you want to know what happened in a run, these are the canonical “truth” artifacts.

#### Stage‑A artifacts

- `outputs/pools/<input>__pool.parquet`
  Pool rows (TFBS-level) with fields such as:
  - scoring: `best_hit_score`, `rank_within_regulator`, `tier`
  - core identity: `tfbs_core`
  - FIMO hit metadata: `fimo_start`, `fimo_stop`, `fimo_strand`, `fimo_matched_sequence` (when captured)
  - selection metadata (MMR): `selection_rank`, `selection_utility`, `nearest_selected_similarity`, etc.

- `outputs/pools/pool_manifest.json`
  Stage‑A sampling truth, including:
  - eligibility rule, tier scheme, FIMO threshold
  - background source (motif background vs bgfile)
  - tier boundary scores and yield counters (generated / candidates_with_hit / eligible_raw / eligible_unique / retained)
  - tier-target success/shortfall reporting
  - PWM consensus string (`pwm_consensus`) and its log‑odds max (`pwm_max_score`)
  - core diversity summaries (k=1 and k=5 nearest‑neighbor distances plus **pairwise weighted‑Hamming**
    distribution, baseline vs actual), overlap, and candidate‑pool diagnostics computed on `tfbs_core` only;
    baseline uses the same candidate slice considered by selection (tier slice/shortlist for MMR).
    Pairwise distances are exact for retained sets; k‑NN distances are deterministically subsampled to 2500
    sequences; entropy uses the full baseline/actual sets; score quantiles are normalized by `pwm_max_score`
    for tradeoff audits; a greedy max‑diversity upper bound (`upper_bound`) is recorded to show whether
    diversity headroom exists in the pool; ΔJ (MMR objective gain) is recorded alongside pairwise Δdiv.
  - mining saturation audit (`mining_audit`) with tail slope Δunique/Δgen to flag plateauing yield
  - padding audit stats (best‑hit overlap with intended core; core‑offset histogram)

Optional Stage‑A debug:

- `outputs/pools/candidates/` (when `keep_all_candidates_debug: true`)
  Candidate-level Parquet logs with accept/reject reasons (`reject_reason`) for audits.

Stage‑A build‑pool stdout:

- Live progress table (screen mode) reports: motif, phase, generated/limit, eligible_unique/target,
  tier yield (0.1/1/9), batch, elapsed, and rate.
- The sampling recap is compact by default (generated, eligible_unique, retained, tier fill, selection,
  pool headroom, diversity delta, overlap, Δscore_norm med, score/length stats).
- Use `dense stage-a build-pool --verbose` to include full diagnostics
  (has_hit, eligible_raw, tier target, set_swaps, Δscore_norm p10).

#### Stage‑B artifacts

- `outputs/libraries/library_builds.parquet` + `outputs/libraries/library_members.parquet`
- `outputs/tables/attempts.parquet`
- `outputs/meta/run_manifest.json` + `outputs/meta/events.jsonl`

Plots that map cleanly to the above:

- `stage_a_summary` — pool quality, yield, tiering, selection, and core diversity diagnostics
- `stage_b_summary` — library feasibility/utilization and solver interaction

`stage_a_summary` requires the diversity block in `pool_manifest.json` and will fail fast if it is missing.

---

### Common footguns and how to avoid them

1) **MMR requires enough eligible unique candidates to fill `n_sites`**
   Symptom: Stage‑A fails (or cannot fill) when `selection.policy: mmr` but eligible unique count is too small
   (often after `uniqueness.key: core` collapses many flank variants).
   Fixes:
   - reduce `n_sites`
   - increase mining (`max_candidates`, `max_seconds`, and/or relax `target_tier_fraction`)
   - switch to `selection.policy: top_score` for small pools
   - if flank diversity matters, consider `uniqueness.key: sequence` (larger unique set)

2) **Core uniqueness requires a core**
   Symptom: core-dedupe can’t work if `tfbs_core` can’t be derived.
   Cause: core identity comes from FIMO `matched_sequence`.
   Fix: `include_matched_sequence` is required for PWM sampling; config validation rejects false.

3) **Tier targeting can be mathematically impossible under your caps**
   With `target_tier_fraction = f`, you need `ceil(n_sites / f)` eligible uniques.
   If you hit `max_candidates` / `max_seconds` first, DenseGen will retain the best available set but it will
   “spill” beyond the target tier (and this will be recorded in the manifest).
   Fixes: increase caps, relax `f`, or reduce `n_sites`.

4) **Regulator labels must match the pool**
   Symptom: constraints like `required_regulators` fail because labels don’t match.
   Fix: use the exact `tf` labels present in Stage‑A pools (for PWM inputs, this is the PWM motif ID).
   Confirm via `dense stage-a build-pool` output or `outputs/pools/pool_manifest.json`.

5) **`cover_all_regulators: true` + sparse pools fails fast (by design)**
   If any regulator has 0 sites after Stage‑A, coverage is impossible.
   Fixes: improve Stage‑A yield (mine more, relax constraints), or use incomplete coverage for exploration
   (`allow_incomplete_coverage: true`) when that is acceptable.

6) **Library size and feasibility interact with fixed elements**
   Symptom: solver repeatedly fails even though Stage‑A pools look fine.
   Cause: fixed elements + required regulators may not fit inside `generation.sequence_length`.
   Fix: increase `sequence_length`, reduce fixed/required burden, or use the feasibility fields in
   Stage‑B artifacts/plots to confirm the bp accounting.

---

### Related references
- [Config field names](../reference/config.md)
- [Stage‑A input types](inputs.md)
- [Constraints + runtime controls](generation.md)
- [Outputs and plots](../reference/outputs.md)

@e-south
