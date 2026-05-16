## RT variant generation

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Builds multi‑mutation RT variants from **single‑amino‑acid** DMS results and quantifies interaction effects:

1. Keep the **best strictly positive** single alternate at each residue.
2. Optionally **restrict positions** (via allow/exclude lists; a “window” can be expressed as an allowed range).
3. Form **non‑colliding** multi‑edit combos for $k \in [k_{\min}, k_{\max}]$ (here, $2..14$).
4. Convert AA edits to DNA using a **codon policy** (e.g., most‑used codon per AA for the organism), preserving per‑base case.
5. Evaluate the **observed** model score on the combined sequence.
6. Compute **epistasis = observed − additive**, where **additive** is the sum of the selected single‑mutation scores.


**Inputs**

* `refs.csv`: `ref_name`, coding DNA `sequence` (length divisible by 3), optional `protein`.
* Singles DMS parquet: round‑1 rows with `permuter__aa_pos`, `permuter__aa_wt`, `permuter__aa_alt`, and a metric (e.g., `permuter__metric__llr_mean`).
* Codon usage CSV: `codon`, `amino_acid`, and `frequency|fraction` (used to select the top/weighted codon).


**Selection**

* Keep round‑1, non‑null; uppercase AAs; **average duplicate** `(pos, wt, alt)`.
* Enforce **positivity**: keep scores `≥ 1e−12` (strictly positive).
* Apply **position filters** (allowed/excluded positions or ranges) and **explicit mutation excludes** if provided.
* **Per‑position best**: for each residue keep the highest‑scoring alt; sort and keep the top N positions (e.g., `25`).
* **Errors**: empty elite set → error; any negative winners with `disallow_negative_best=True` → error. (Zeros are filtered out by `min_delta`.)

**Combination**

* **Strategy: `enumerate` (exhaustive, deterministic)** — emit **every** unique, non‑colliding combo for each $k \in [2..14]$.

  * For $p$ selected positions, total variants = $\sum_{k=2}^{14} \binom{p}{k}$.
  * **Note:** `budget_total` is **ignored** in `enumerate`.
* (Alternative) `random`: sample under `budget_total` with optional per‑k targets.

**DNA modifications**

* **Codon policy:** `top` (most‑used codon) or `weighted` (by usage).
* **Case preservation:** when swapping a 3‑nt codon, maintain original per‑base upper/lower case.
* Emit an audit trail in `modifications` (combo header, AA tokens, nucleotide‑change tokens).

**Scoring & outputs (canonical columns)**

* **Additive expectation:** `permuter__expected__llr_mean = sum(single scores)`
* **Observed (after evaluation of the mutli-site seqeunce):** `permuter__observed__llr_mean`.
* **Epistasis (post‑eval):** `epistasis = permuter__observed__llr_mean − permuter__expected__llr_mean`.
* Also emitted per row: `sequence`, `aa_combo_str` (position‑sorted), `aa_pos_list`, `aa_wt_list`, `aa_alt_list`, `mut_count`, `proposal_score`, `round=2`, and `modifications`.


**Hard errors (no silent fallbacks)**

* Missing required files/columns; reference length not multiple of 3; WT AA mismatch; empty selection; disallowed negatives.


**Artifacts**

* Parquet containing the fields above; plots for ranking, epistasis scatter, and metric vs. mutation count.
