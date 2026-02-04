## Codex punch list (highest impact first)

### P0 — Determinism gap: auto‑opt bootstrap CI seed is **not deterministic across runs**

**Narrative expectation:** “Bootstrap CI on top‑K median (deterministic seed from pilot metadata).”
**Reality:** `_bootstrap_seed()` includes `run_dir.name`, which embeds timestamps/hashes ⇒ same config can yield different CI bounds across reruns, and **may flip “confident vs not confident”** near ties.

**Where**

* `src/dnadesign/cruncher/src/app/sample_workflow.py`

  * `_bootstrap_seed()`
  * `_aggregate_candidate_runs()` (its pooled bootstrap seed also depends on run dir names)

**Fix**

1. Make bootstrap seeding depend only on stable metadata (seed, attempt, candidate, budget, replicate, length, regulator set), not filesystem run names.
2. For pooled (replicate-aggregated) CI, base the seed on stable per-replicate metadata (e.g., each replicate’s `manifest["seed"]` + `(budget, kind, length)`), not `run_dir.name`.

**Suggested implementation (patch sketch)**

* In `_bootstrap_seed()` remove `"run": run_dir.name`.
* Add more stable disambiguators (optional but good): `sequence_length`, `regulator_set` TF list.

```python
def _bootstrap_seed(*, manifest: dict[str, object], run_dir: Path, kind: str) -> int:
    auto_meta = manifest.get("auto_opt") or {}
    payload = {
        "seed": manifest.get("seed"),
        "kind": kind,
        "attempt": auto_meta.get("attempt"),
        "candidate": auto_meta.get("candidate"),
        "budget": auto_meta.get("budget"),
        "replicate": auto_meta.get("replicate"),
        "length": manifest.get("sequence_length"),
        "tfs": (manifest.get("regulator_set") or {}).get("tfs"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)
```

* In `_aggregate_candidate_runs()`, derive a pooled seed from **stable replicate seeds**:

```python
rep_seeds = []
for run_dir in run_dirs:
    m = load_manifest(run_dir)
    rep_seeds.append(m.get("seed"))
payload = {"kind": kind, "length": length, "budget": budget, "rep_seeds": sorted(rep_seeds)}
seed = int(hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:8], 16)
```

**Acceptance criteria**

* With `sample.rng.deterministic=true`, rerunning auto‑opt with identical config yields identical CI bounds and identical “confident vs not confident” decisions (assuming identical scores).
* CI seed does not change if run directory names change.

**Tests**

* Unit test: same `manifest` + different `run_dir.name` ⇒ same seed.
* Unit test: pooled seed stable given same replicate seeds (independent of run dir names).

---

### P0 — dsDNA equivalence is **optional + fragmented**, creating intent drift + misleading “unique_fraction”

**Narrative expectation:** “Scan both strands (dsDNA), treat reverse complements as equivalent for uniqueness/diversity.”
**Reality:**

* Strand scanning (`objective.bidirectional`) is separate from diversity/uniqueness equivalence:

  * diversity filter uses dsDNA only if `sample.elites.dsDNA_hamming=true`
  * elites canonicalization only if `sample.elites.dsDNA_canonicalize=true`
* `sequences.parquet` lacks canonical sequence, so downstream diagnostics like `unique_fraction` can silently treat reverse complements as distinct (depends on `summarize_sampling_diagnostics` implementation, but highly likely).

**Where**

* `src/dnadesign/cruncher/src/app/sample_workflow.py`

  * writing `sequences.parquet` (no canonical column)
  * elite selection uses dsDNA hamming optionally (good), but the overall UX is confusing
* `src/dnadesign/cruncher/src/core/optimizers/gibbs.py` and `pt.py`

  * internal elite selection uses forward Hamming (`np.sum(seq_arr != e.seq)`), ignoring dsDNA setting

**Fix (minimal + high ROI)**

1. Add `canonical_sequence` column to `sequences.parquet` when dsDNA canonicalization is enabled (or when `objective.bidirectional` is true and user wants dsDNA equivalence).
2. Update diagnostics to prefer canonical uniqueness if that column exists.
3. Plumb `dsdna_hamming` into optimizer config so internal elite selection and any optimizer-reported elite metadata is consistent (even if sample_workflow later reselects elites).

**Concrete steps**

* In `_run_sample_for_set()` when writing sequences rows:

  * if `sample_cfg.elites.dsDNA_canonicalize` add `"canonical_sequence"` per row using `canon_int(seq_arr)`.

```python
want_canonical_all = bool(sample_cfg.elites.dsDNA_canonicalize)

# inside _sequence_rows()
if want_canonical_all:
    row["canonical_sequence"] = SequenceState(canon_int(seq_arr)).to_string()
```

* Pass `dsdna_hamming` into `opt_cfg`:

```python
opt_cfg["dsdna_hamming"] = bool(sample_cfg.elites.dsDNA_hamming)
```

* In `GibbsOptimizer` / `PTGibbsOptimizer`:

  * read `self.dsdna_hamming = bool(cfg.get("dsdna_hamming", False))`
  * use `dist_fn = dsdna_hamming if self.dsdna_hamming else hamming_distance` for elite selection at end.

**UX improvement**

* Add a warning (or auto-default behavior) when `objective.bidirectional=true` but dsDNA equivalence is off:

  * “bidirectional scoring is enabled but dsDNA_hamming/dsDNA_canonicalize are disabled; reverse complements will be treated as distinct for diversity/uniqueness.”

**Acceptance criteria**

* When dsDNA canonicalization is enabled, reverse complements collapse to one canonical identity in sequences output and diagnostics.
* Optimizer-returned elites respect dsDNA distance if enabled.

**Tests**

* Unit test: `canon_int(seq)` equals `canon_int(revcomp(seq))`
* Pipeline test: if `canonical_sequence` exists, `unique_fraction` computed from canonical IDs (in diagnostics module)

---

### P1 — Performance win: elite filtering rescans every sequence even when score_scale is `normalized-llr`

**Reality:** In `_run_sample_for_set()`, elite candidate loop calls:

```python
norm_map = scorer.normalized_llr_map(seq_arr)
```

That rescans windows for every sample across every TF. In the demo config, `score_scale: normalized-llr`, meaning **the optimizer already computed those exact normalized values** into `per_tf_map`.

**Where**

* `src/dnadesign/cruncher/src/app/sample_workflow.py` elite selection loop

**Fix**

* If `sample_cfg.objective.score_scale == "normalized-llr"`, reuse `per_tf_map` as `norm_map` (or build a filtered dict in TF order), skipping `normalized_llr_map()`.

```python
use_norm_from_scores = (sample_cfg.objective.score_scale.lower() == "normalized-llr")

for (...), seq_arr, per_tf_map in ...:
    if use_norm_from_scores:
        norm_map = {tf: float(per_tf_map.get(tf, 0.0)) for tf in scorer.tf_names}
    else:
        norm_map = scorer.normalized_llr_map(seq_arr)
```

**Why it matters**

* This can be a dominant runtime cost for large budgets because it’s O(samples × TFs × windows) and is completely avoidable for the common `normalized-llr` case.

**Acceptance criteria**

* For `score_scale=normalized-llr`, elite selection does not call `normalized_llr_map()` and results are identical to previous behavior.

**Tests**

* Unit test with monkeypatch: make `scorer.normalized_llr_map` raise; ensure no raise when score_scale is normalized-llr.
* Regression test: elite set unchanged for normalized-llr runs.

---

### P1 — Alignment gap: “min/soft‑min across TFs” is **not universally true** (`consensus-neglop-sum` silently switches to sum)

**Narrative expectation:** combined objective is min/soft-min across TFs.
**Reality:** `SequenceEvaluator` hard-special-cases:

* If `scale == "consensus-neglop-sum"` ⇒ combiner defaults to `sum()` and `_use_softmin=False`.

**Where**

* `src/dnadesign/cruncher/src/core/evaluator.py` (`SequenceEvaluator.__init__`)
* `src/dnadesign/cruncher/src/app/sample_workflow.py` sets combiner=sum for that scale

**Two valid resolution paths (pick one explicitly)**

#### Option A (doc‑only; minimal code)

* Update docs + “audit-ready narrative” to state:

  * `consensus-neglop-sum` uses **sum across TFs**, not min, and soft-min is disabled.
* Add explicit logging in sample_workflow:

  * “Objective combiner = sum (scale=consensus-neglop-sum); weakest-TF optimization is not active.”

#### Option B (code + UX; more robust)

* Decouple **scale** from **combiner** by adding an explicit config field:

  * `objective.combine: "min" | "softmin" | "sum"` (or `"min"|"sum"` + keep `softmin` schedule)
* Default combine to `"min"` to match narrative, regardless of scale.
* Keep backward compatibility:

  * if `score_scale == "consensus-neglop-sum"` and combine not specified, default combine to `"sum"` but emit a warning that this deviates from weakest‑TF objective.

**Acceptance criteria**

* Users can choose `sum` vs `min` explicitly.
* Manifest and logs state the combiner used.

**Tests**

* For same per-TF values, ensure `combined(min)` != `combined(sum)` and the chosen mode is reflected in manifest.

---

### P2 — UX: chain indexing + draw indexing are internally consistent but user-hostile

**Reality**

* `chain` stored 0-based in parquet + JSON/YAML (but status UI logs and best_chain are 1-based).
* `draw` for draw-phase starts at `tune` (absolute sweep index), not at 0.

This is fine for internal bookkeeping but creates easy confusion when users plot or filter.

**Where**

* `src/dnadesign/cruncher/src/app/sample_workflow.py`

  * sequences.parquet rows use `chain_id` as-is
  * elites entries use `cand.chain_id` as-is
  * draw indexes are absolute sweep indices

**Fix (additive, non-breaking)**

* Add explicit columns/fields:

  * `chain_1based = chain + 1`
  * `sweep = draw` (rename in output only) OR add `draw_in_phase = draw - tune` for phase=draw
* Add brief note to `elites.yaml` metadata.

**Acceptance criteria**

* Existing analysis keeps working.
* New columns make it trivial to plot with intuitive indices.

---

### P2 — Auditability: persist “final softmin beta used for combined_score_final” explicitly

You already compute `beta_softmin_final` via `_resolve_final_softmin_beta()`, but it’s only implicitly recoverable from optimizer stats and config.

**Where**

* `sample_workflow.py`:

  * sequences.parquet writes combined_score_final
  * elites uses same
  * manifest includes optimizer_stats

**Fix**

* Add to manifest under `objective`:

  * `softmin_beta_final_resolved: <float|null>`
  * optionally `softmin_schedule_summary` (if available from optimizer.objective_schedule_summary())
* Add to `elites.yaml` metadata too.

**Acceptance criteria**

* An auditor can read manifest + parquet and know exactly which beta produced `combined_score_final` without re-deriving schedules.

---

### P2 — PT stats correctness: `final_mcmc_beta()` ignores ladder adaptation scale

**Reality**

* `PTGibbsOptimizer.final_mcmc_beta()` returns `max(self.beta_ladder_base)` even when adaptive swap controller rescales the ladder during sampling.

**Where**

* `src/dnadesign/cruncher/src/core/optimizers/pt.py`

**Fix**

* Return `max(self.beta_ladder)` (the effective ladder), and include both base and effective ladders in stats:

  * `beta_ladder_base`
  * `beta_ladder_final` (effective at end)
  * `beta_ladder_scale_final`

**Acceptance criteria**

* Manifest reflects the actual inverse temperatures used during sampling.

---

## Optional “bigger” performance tracks (only if needed)

### P3 — Reduce evaluation cost of “S” moves (currently 4 full rescans per single-base update)

**Reality**

* In both `gibbs.py` and `pt.py`, `S` moves compute `evaluator.evaluate(...)` four times (one per base).
* Each evaluate rescans all PWMs and all offsets (and both strands), dominating runtime.

**High-level approach**

* Maintain per-PWM window scores and current best window index/score per TF.
* On a single-base change at position `i`, only windows whose start is in `[i-w+1, i]` are affected.
* Update affected window scores incrementally and update best score cheaply.

**Implementation note**

* This is the kind of change that’s worth gating behind a feature flag (e.g., `optimizer.fast_eval=true`) until validated.

**Acceptance criteria**

* Same stationary distribution (or at least comparable optimization behavior) with significantly lower CPU time per sweep.
* Property tests: incremental evaluator matches full evaluator for random sequences/moves.

---

## Quick summary of “expected vs real” gaps (so you can decide code vs docs)

1. **Bootstrap CI determinism:** expected deterministic; real seed depends on run dir name.
2. **dsDNA equivalence:** expected default/holistic; real behavior split across `bidirectional`, `dsDNA_hamming`, `dsDNA_canonicalize`, and likely diagnostics uniqueness.
3. **Objective = min/soft-min across TFs:** expected universal; real `consensus-neglop-sum` uses sum and disables soft-min.
4. **Performance:** elite gating does redundant rescans even when score_scale already provides normalized values.

If you want the most leverage with lowest risk: do P0 + the P1 normalized-llr reuse first (they’re largely additive and won’t change the optimizer’s core semantics).
