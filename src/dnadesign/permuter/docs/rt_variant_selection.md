
### Multi‑site mutant variant selection

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-24

This is a historical RT multi-site selection note refreshed for the current
workspace and metric-column contracts. Treat the concrete embedding sizes and
cluster labels below as workflow-specific inputs, not Permuter core API
requirements.

From ~16k multi‑mutant RT variants (mutated at $k = 2,\dots,14$ positions) scored with Evo‑2 $LLR$ and embedded as mean‑pooled logits ($V=512$), we want to pick a subset to synthesize such that:

1. **All selected variants are strongly epistatic**
   – i.e. large **positive** epistasis (observed − expected additive baseline).

2. **Within that high‑epistasis band**, the set is **diverse**:

   * spread out in the Evo‑2 logits latent space (angular distance),
   * and not dominated by any single pre‑computed cluster.

We treat this as a **score‑gated, diversity‑constrained selection problem**:

* First, build a **composite score**.
* Second, restrict to a **high‑score candidate pool**.
* Third, within that pool, perform **diversity‑aware selection** in embedding space, with optional cluster caps.

### Data inputs

From the source multi‑mutant dataset (`records.parquet`, e.g. from a `combine_aa` run followed by `evaluate`):

* **Observed fitness**

  * `permuter__observed__llr_mean` (or a similar LLR metric)
* **Expected additive baseline**

  * `permuter__expected__llr_mean`
* **Epistasis (synergy)**

  * `permuter__interaction__epistasis__<metric>` (computed upstream as observed − expected)
* **Logits embedding**

  * `permuter__observed__logits_mean` → `list<item: double>` (length 512)
* **Combo metadata (AA positions 1‑indexed)**

  * `permuter__aa_pos_list`: `list<int64>`
  * (optional) `permuter__aa_wt_list`, `permuter__aa_alt_list`
  * `permuter__aa_combo_str`: string like `"G16F|L17I|N21H|…"`
  * `permuter__mut_count`: `int64` (alias (k))
* **Cluster labels**

  * `select.clusters.column` names the upstream cluster-id column.
  * Use `column: null` only for explicit no-cluster selection; cluster caps and cluster quality filters must be disabled in that mode.
* **Proposal info**

  * `permuter__proposal_score` (optional scalar)
* **USR core columns** (from `run.py`)

  * `id`, `sequence`, `bio_type`, `alphabet`, `length`, `source`, `created_at`
* **Optional sidecars**

  * `REF.fa` (reference DNA)
  * `REF_AA.fa` (reference protein)

### Observed fitness

Let $\mathrm{LLR}_\text{obs}(v)$ be the log‑likelihood ratio of variant $v$ relative to the reference:

$$
\mathrm{LLR}_\text{obs}(v)
= \log P(\text{variant}) - \log P(\text{ref}),
$$

with $\mathrm{LLR}_\text{obs}(\text{ref}) \approx 0$ by construction.

### Expected additive baseline from singles

Upstream (e.g. in `combine_aa`), we define an additive baseline by summing single‑mutant effects over the $k$ mutated positions:

$$
\mathrm{LLR}_\text{exp}(v)
= \sum_{i \in \text{mutated sites of } v} \mathrm{LLR}_{\text{single}, i}.
$$

This is stored as `permuter__expected__<metric>`.

### Epistasis (synergy)

$$
\Delta(v) = \mathrm{LLR}_\text{obs}(v) - \mathrm{LLR}_\text{exp}(v).
$$

* $\Delta > 0$: **synergistic** (higher than additive expectation).
* $\Delta < 0$: **antagonistic** (worse than additive).

For multi‑site selection, we only consider **non‑negative** epistasis and then favor variants with **large positive** $\Delta$.
Permuter stores this as `permuter__interaction__epistasis__<metric>`, not as a
bare `epistasis` column.

### Embedding geometry (angular distance)

Let $u, v \in \mathbb{R}^V$ be mean‑pooled Evo‑2 logits for two variants. We L2‑normalize:

$$
\hat u = \frac{u}{\lVert u \rVert_2}, \qquad
\hat v = \frac{v}{\lVert v \rVert_2},
$$

define cosine similarity $s(u, v) = \langle \hat u, \hat v \rangle$, and **angular distance**:

$$
d_\angle(u, v) = \arccos\big(\operatorname{clip}(s(u, v), -1, 1)\big)
\quad\in [0,\pi].
$$

This is the primary notion of “semantic distance” used for diversity.

### Pairwise distances for diagnostics

Used only in diagnostic plots (not for selection itself):

* **Mutation‑count delta:** $|k_i - k_j|$ for unordered pairs $(i,j)$,
  from `permuter__mut_count`.
* **Levenshtein (AA):** edit distance between full AA sequences
  constructed from `REF_AA.fa` and `permuter__aa_combo_str`
  (Hamming when lengths equal).

## Score normalization

We put LLR and epistasis on comparable scales via robust median/MAD normalization.

Let:

* $\mathrm{LLR}_\text{obs}(v)$ be observed fitness.
* $\Delta(v)$ be epistasis (from `permuter__interaction__epistasis__<metric>`).

In MAD mode (`normalize.method: mad`):

$$
z_{\text{epi}}(v)
= \frac{\Delta(v) - \operatorname{median}(\Delta)}{\operatorname{MAD}(\Delta)},\qquad
z_{\text{llr}}(v)
= \frac{\mathrm{LLR}_\text{obs}(v) - \operatorname{median}(\mathrm{LLR}_\text{obs})}
{\operatorname{MAD}(\mathrm{LLR}_\text{obs})}.
$$

* `gaussian_consistent: true` multiplies MAD by 1.4826.
* `gaussian_consistent: false` uses raw MAD.

### k‑aware scaling (`stratify_by_k`)

The yaml knob `select.scoring.normalize.stratify_by_k` can be:

* `off`

  Use global robust scaling for all k.

Remove any other option, remove k-aware scaling altogether.

### Composite per‑variant score

We define a composite score using user‑configurable weights:

$$
\text{score}(v)
= \alpha \cdot z_{\text{llr}}(v) + \beta \cdot z_{\text{epi}}(v),
$$

where:

* $\alpha = \texttt{select.scoring.weights.llr}$,
* $\beta = \texttt{select.scoring.weights.epi}$.

For example, in the the reverse transcriptase selection scope:

* $(\alpha,\beta) = (0.0, 1.0)$, i.e. **pure epistasis‑driven score**.

This makes `score` a strictly increasing function of epistasis: higher positive $\Delta$ ⇒ higher `score`.

### Candidate filtering & score‑gated pool

To guarantee that **all selected variants are strongly scored**, we apply a two‑step constraint:

1. **Hard validity filter** (row‑level)
2. **Score‑based gating** (pool construction)

### 1. Hard validity filter

We first drop any row that fails:

* **Finite LLR:** `permuter__observed__*` (the LLR column chosen above) is non‑null and finite.
* **Numeric, non‑negative epistasis:**

  * `epistasis` coerces to `float64`,
  * not NaN,
  * ≥ 0.0 (we exclude antagonistic variants by design).
* **Valid AA positions:**

  * `permuter__aa_pos_list` is present and non‑null,
  * parses to a non‑empty list of integer positions.
* **Valid embedding:**

  * `permuter__observed__logits_mean` is a numeric 1D vector
    of consistent length across rows with only finite values.

Only rows passing all checks enter the scoring pipeline. Clear summary statistics of the proportions of rows accepted are emitted as logging info.

### 2. Score‑based gating (high‑epistasis pool)

We then **sort the valid rows by composite score** (descending), using a deterministic tie‑breaking order, and keep only the top slice as a **high‑score candidate pool**.

Tie‑breaking order (from strongest to weakest):

1. **Composite score** (`score`) — higher is better.
2. **Mutation count** (`permuter__mut_count`) — smaller is better
   (if `tie_breakers.prefer_fewer_mutations: true`).
4. **Variant id** (`permuter__var_id`) — lexicographically ascending
   (final stable tie‑breaker if present).

Using this ordering, we define:

* `total_variants` = `select.budget.total_variants`
* **Score pool factor** $f_\text{pool} \ge 1$.

Then:

$$
\text{pool_size} = \min\big(\text{len(valid)}, \max(\text{total_variants}, f_\text{pool} \cdot \text{total_variants})\big)
$$

and

* Let **`df_pool`** be the first `pool_size` rows from the tie‑broken, score‑sorted list.

All later diversity decisions are restricted to `df_pool`. This guarantees:

### Cluster representation & quality filters

Clusters provide a coarse semantic grouping derived from an upstream column named by `select.clusters.column`.
The historical RT workflow used `cluster__perm_v1`; the workspace-backed `combine_aa` output has no cluster labels, so `rt_multisite_select` declares `column: null`.

For each cluster $c$, we compute:

1. **Medoid (for diagnostics)**

   Using only rows in `df_pool` that belong to cluster $c$, we take:

   $$
   m_c
   = \arg\min_{x \in c}
   \frac{1}{|c|}
   \sum_{y \in c} d_\angle(x, y),
   $$

   i.e. the vector whose average angular distance to other members is minimal. The corresponding row index is stored as `medoid_row`.

2. **Cluster‑level statistics** (written to `CLUSTER_SUMMARY.parquet`)

   * `cluster_id`
   * `size` (number of candidate rows for that cluster in `df_pool`)
   * `mean_z_llr`, `mean_z_epi`, `mean_composite`
   * `pos_epi_fraction`: fraction with `epistasis > 0`
   * `loc_stat`: mean / median / trimmed mean of `z_llr`, per `select.clusters.filters.location_stat`

3. **Per‑cluster capacity (optional)**

   The knob:

   ```yaml
   select:
     clusters:
       column: cluster__perm_v1
       picks_per_cluster: K  # e.g. 1 or 2
   ```

   acts as a **soft cap**: at most **K** selected variants from any single configured cluster.
   This requires a real upstream cluster column; it is rejected when `column: null`.

### Diversity‑aware selection within the high‑score pool

Once we have:

* a filtered candidate pool `df_pool` (high‑score variants only),
* L2‑normalized embeddings (U) for that pool,
* and cluster‑level quality filters applied,

we run a **score‑ordered, diversity‑filtered selection**.

### Embedding preparation

Let:

```python
emb_pool = extract_embedding_matrix(df_pool[embedding_col])
U = l2_normalize_rows(emb_pool)  # [N_pool, D]
```

Each row $U_i$ is a unit vector in logits space.

### Diversity knob: minimum angular separation

From YAML:

```yaml
select:
  budget:
    total_variants: 100
    intracluster_diversity:
      enabled: true
      min_angular_distance_deg: 6.0
```

When `intracluster_diversity.enabled: false` or
`min_angular_distance_deg: 0.0`, the angular diversity constraint is
effectively disabled.

### Selection algorithm (score‑first, diversity‑second)

We iterate over `df_pool` **in descending score order** and build the selected set greedily:

1. Initialize:

   * `selected = []` (indices into `df_pool`)
   * `selected_clusters = {}` → cluster id → count
   * `selected_vectors = []` (rows of $U$ for selected variants)

2. For each candidate row `i` in `df_pool` (in score order):

   a. If `len(selected) == total_variants`: **stop** (budget filled).

   b. Let `cluster_id = df_pool[select.clusters.column][i]`, or `all` when `select.clusters.column: null`.

   * If `picks_per_cluster` is set and
     `selected_clusters.get(cluster_id, 0) >= picks_per_cluster`, **skip** this candidate
     (cluster capacity reached).

   c. **Diversity check** (if `intracluster_diversity.enabled: true` and we have at least one selected variant):

   * Compute angular distance from this candidate to each already selected variant:

     $$
     \theta_\text{min}(i)
     = \min_{j \in \text{selected}} d_\angle(U_i, U_j).
     $$

   * If $\theta_\text{min}(i) < \theta_\text{min}$, **skip** this candidate
     (too similar in embedding space to existing picks).

   d. Otherwise, **accept** the candidate:

   * Append `i` to `selected`.
   * Increment `selected_clusters[cluster_id]`.
   * Append `U_i` to `selected_vectors`.

3. If we reach the end of `df_pool` with fewer than `total_variants` picks
   (e.g. because `theta_min` was too strict), we accept that the budget is
   under‑filled.

   * either accept that the budget is under‑filled, or
   * optionally relax `theta_min` in a second pass (implementation choice; document in logs).

**Key property:**

> Because we iterate in **strict score order** and only ever **skip** candidates that fail diversity / cluster caps, every selected variant is both:
>
> * high‑scoring (within the top score‑gated pool), and
> * not too close in logits space to any previously selected variant.

### Final picks and artifacts

The selection step returns:

* the indices of selected rows within `df_pool` (and thereby within the original dataset),
* a selected table `picks_df` with the following fields (at minimum):

  * `source_id` (original `id` from the input dataset),
  * `sequence` (canonical sequence; for `MULTISITE_SELECT.*` we may also include a decorated version with mutated codons uppercased),
  * `cluster_id` (from `select.clusters.column`, or `all` in explicit no-cluster mode),
  * `k` (`permuter__mut_count`),
  * `llr_obs`, `llr_exp`, `delta` (= epistasis),
  * `z_llr`, `z_epi`, `score`,
  * `aa_pos_list`, `aa_combo_str`,
  * `proposal_score` (if present),
  * `angle_to_nearest_selected_medoid_deg` (optional diagnostic),
  * `angle_to_cluster_medoid_deg` (optional diagnostic).

**Outputs:**

* **`MULTISITE_SELECT.parquet` / `.csv`**

  * Selected variants with the enriched schema above.
* **`CLUSTER_SUMMARY.parquet`**

  * Per‑cluster stats for all clusters present in `df_pool`, including:

    * `cluster_id`, `size`, `mean_z_llr`, `mean_z_epi`, `mean_composite`,
    * `pos_epi_fraction`,
    * `loc_stat`,
    * `medoid_row`,
    * for clusters that actually contributed picks: `min_inter_medoid_angle_deg`.
* **`SELECT_SUMMARY.md`**

  * Human‑readable summary:

    * total candidates, total picks,
    * score pool size and thresholds,
    * histogram of mutation counts,
    * per‑cluster pick counts and best/mean scores,
    * global minimum pairwise angle among selected variants.
* **Diagnostic figures**

  * Mutation count distributions, pairwise |Δk|, pairwise AA Levenshtein, and HEB (see below).

---

## Diagnostic plot specifications

1. **Mutation‑count histogram (selected vs random)**

   * **Filename:** `fig_mut_count_hist_selected.png`
   * **What:** Distribution of `permuter__mut_count` for **Selected** vs a random background sample from the filtered candidate pool (matched in size or scaled by `diagnostics.random_sample_factor`).
   * ***Axes:** x = mutation count $k$, y = count.

2. **Pairwise |Δk| — selected vs random**

   * **Filename:** `fig_pairwise_delta_k_selected_vs_random.png`
   * **Computation:** For each set, compute $|k_i - k_j|$ for all unordered pairs.
   * **Curves:** KDE for **Selected** vs random background.
   * **Axes:** x = $|Δk|$, y = pairwise count.

3. **Pairwise Levenshtein (AA) — selected vs random**

   * **Filename:** `fig_pairwise_levenshtein_selected_vs_random.png`
   * **Computation:** Build full‑length AA sequences from `REF_AA.fa` and `aa_combo_str`; compute Levenshtein distance for all unordered pairs; repeat for random background.
   * **Curves:** KDE for **Selected** vs random.
   * **Axes:** x = Levenshtein distance, y = pairwise count.

4. **Hierarchical Edge Bundling (HEB) — selected only**

   * **Filenames:**

     * `fig_edge_bundling_selected.png` (PNG),
     * optionally `fig_edge_bundling_selected.pdf` (vector) when `heb.out_svg: true`.
   * **Nodes:** unique AA mutation tokens across selected variants (e.g., `G16F`, `L17I`, …).
   * **Edges:** between tokens (a,b) that co‑occur in at least one selected variant.
   * **Edge width:** $\propto \sqrt{\text{co‑occurrence count}}$ by default.
   * **Color mapping (`heb.color_by`):**

     * `node_avg_k`: node hue encodes average mutation count among variants containing each token.
     * `edge_avg_k`: edge hue encodes average (k) over variants that contain both tokens.
   * **Exports:** `EDGE_BUNDLE_TABLE.parquet` with columns `{token_a, token_b, weight_count, avg_k}`.

### Overall algorithm

Putting it all together:

1. **Load & validate source dataset**

   * Confirm presence of the required columns and sidecars.
   * Drop rows with:

     * missing / invalid LLR or epistasis,
     * epistasis < 0,
     * missing AA position list,
     * invalid `logits_mean` embedding.

2. **Robust scaling & composite score**

   * Compute robust z‑scores `z_llr`, `z_epi`, optionally k‑aware.
   * Compute composite `score = α z_llr + β z_epi` using YAML weights (RT scope: α=0, β=1).

3. **High‑epistasis pool (score gating)**

   * Sort valid rows by `score` (descending) with deterministic tie‑breakers.
   * Take top `pool_size` rows as `df_pool` (e.g. `pool_size ≈ 3 × total_variants`).

4. **Cluster summarization & filters**

   * For each configured cluster present in `df_pool`:

     * compute medoid, `mean_z_llr`, `mean_z_epi`, `mean_composite`,
       `pos_epi_fraction`, and `loc_stat`;
     * apply `min_cluster_mean_z_llr` and `min_cluster_pos_epistasis_fraction`
       if configured; reject clusters that fail.
   * Restrict `df_pool` to variants in the remaining clusters.

5. **Diversity‑aware selection within `df_pool`**

   * Prepare L2‑normalized embeddings.
   * For variants in **score order**, greedily:

     * enforce **per‑cluster caps** (`picks_per_cluster`) if set;
     * enforce **minimum angular separation** (`intracluster_diversity.min_angular_distance_deg`)

6. **Emit artifacts**

   * Write `MULTISITE_SELECT.parquet` / `.csv`.
   * Write `CLUSTER_SUMMARY.parquet` and `SELECT_SUMMARY.md`.
   * Generate diagnostic figures and HEB plots (if enabled).
   * Append a `SELECT` entry to `RECORD.md` summarizing key knobs and counts.

**Guarantees:**

* **Epistasis:**
  All selected variants have non‑negative epistasis, and are drawn from a **score‑gated high‑epistasis pool**.

* **Score priority:**
  Selection order respects the composite score; diversity can only *disqualify* a high‑score candidate if it is too similar in embedding space (or violates cluster caps). No lower‑score candidate can pre‑empt a higher‑score candidate that passes diversity checks.

* **Diversity:**
  Selected variants are **well separated in logits space** (by at least `min_angular_distance_deg` when enabled), and optionally spread across clusters via `picks_per_cluster` and cluster filters.

---

### CLI usage

Example invocation:

```bash
permuter run \
  --workspace rt_multisite_select \
  --ref retron_Eco1_RT_wt
```

This:

* Generates a new dataset for the selected multi‑mutants,
* Writes the selection artifacts (`MULTISITE_SELECT.*`, `CLUSTER_SUMMARY.parquet`,
  `SELECT_SUMMARY.md`, diagnostics),
* Appends a structured `SELECT` section to the dataset’s `RECORD.md`.

---

### Logging / UX expectations

During a run, logs should clearly report:

* **Data:**
  Number of input variants, number retained after validation; counts by $k$ and drops by cause (NaN epistasis, negative epistasis, missing embeddings, etc.).

* **Scaling / scoring:**
  Medians and MADs for LLR and epistasis; whether k‑aware scaling was enabled; composite weight tuple (α, β).

* **Score pool:**
  `pool_size`, score range of the pool vs full dataset; summary of how much of the epistasis tail is retained.

* **Clusters:**
  Number of clusters pre‑ and post‑filter; which clusters were dropped by quality filters; per‑cluster pick counts and best/mean scores.

* **Diversity:**
  Minimum pairwise angle among selected variants; summary of `min_angular_distance_deg` and whether it limited the budget.

* **Diagnostics:**
  Size and construction of random comparator samples; paths to saved figures and HEB tables.

---

@e-south
