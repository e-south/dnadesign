## DenseGen Improvement Plan

This document captures the current critical evaluation of DenseGen, a prioritized improvement plan,
and Codex-ready prompt packs to drive concrete patches. It is engineering-first: correctness > auditability
> UX > performance.

### Contents
- [Critical evaluation of DenseGen as it stands](#critical-evaluation-of-densegen-as-it-stands)
- [High-ROI improvements, prioritized](#high-roi-improvements-prioritized)
- [Design decisions to fortify intent](#design-decisions-to-fortify-intent)
- [Codex CLI prompt packs](#codex-cli-prompt-packs)
- [Implementation notes / gotchas](#implementation-notes--gotchas)
- [Recommended prompt order](#recommended-prompt-order)

---

## Critical evaluation of DenseGen as it stands

### What is strong (and worth preserving)

- Strictness is a feature here. Config validation is tight (extra="forbid", strict YAML duplicate-key
  failure, run-scoped path enforcement). That is exactly what you want for reproducible bio-design
  pipelines.
- Separation of concerns is clean enough: sources -> sampling -> optimizer adapter -> postprocess ->
  sinks -> plotting. The adapter boundary around dense_arrays is good.
- Metadata is centralized + schema-validated (build_metadata + metadata_schema.validate_metadata).
  That is high-leverage, and it is why the gaps are clear.
- Pipeline determinism: a single seed drives Python RNG + NumPy RNG and thus PWM sampling + TFBS
  sampling. That is critical for scientific reproducibility.

### What is fragile or incorrect (high priority)

These are the ones to treat as must-fix before trusting downstream analyses:

1. Offset correctness bug when gap fill is 5prime
   used_tfbs_detail.offset is extracted from the pre-padding sequence (sol.sequence). If
   gap_fill_end=5prime, final sequence coordinates shift, but metadata does not. This is a hard
   correctness defect for any downstream indexing/coverage/variant mapping.

2. Promoter motif filtering is incomplete if multiple promoter constraints exist
   _compute_used_tf_info() only inspects pcs[0] to build promoter_motifs. But the adapter can apply
   multiple promoter constraints (it iterates and calls opt.add_promoter_constraints for each).
   That means some fixed motifs may leak into used_tfbs_detail or be inconsistently excluded.

3. Constraint double enforcement can mask underlying problems
   Regulator constraints are pushed into dense_arrays via add_regulator_constraints, then also
   filtered after solving (required_regulators, min_count_by_regulator, min_required_regulators,
   etc.). This is defensible if DenseArrays is approximate or constraints are soft, but it is also
   a smell: it can waste solver time, stall/resample due to metadata interpretation rather than
   solver feasibility, and silently diverge between solver constraints and DenseGen filters.
   If the post-filter remains, record which constraint failed and whether the solver claimed it
   satisfied constraints.

### What is missing for auditability / reproducibility (medium-high priority)

- Global sequence composition is untracked (GC only for padding). This blocks basic QA (composition
  drifts, batch effects, library bias).
- No stable identity for libraries per resample (hard to group or debug why a library stalled).
- No solver provenance / status / timing (cannot compare backends or prove runs are comparable).
- No run-level manifest (metadata is per-record; lacks a compact run summary of resamples/failures).

### What is already better than the report implies (minor correction)

- Side-bias motifs are not excluded from used_tfbs_detail by current code (only promoter motifs and
  motifs beyond orig_n are excluded). If a report claims side-bias motifs are excluded, that is
  incorrect for the current pipeline.

---

## High-ROI improvements, prioritized

Implement in phases aligned with impact and merge risk.

### Phase 0 - Fix correctness and stabilize semantics (do first)

1. Fix offsets under 5prime padding
   - Define semantics: used_tfbs_detail.offset refers to final sequence coordinates.
   - Add offset_raw alongside it for traceability.
   - Add length and end (or end_final) per placement.

2. Fix promoter constraint motif handling
   - Collect upstream/downstream motifs across all promoter constraints, not just the first.

3. Record padding length explicitly in placement entries
   - If gap_fill_used and gap_fill_end=5prime, each placement should record pad_left = gap_fill_bases
     (or 0 otherwise).

### Phase 1 - Auditability: GC + input + provenance

4. Add gc_total (final sequence) and gc_core (pre-pad sol.sequence) top-level metadata fields.

5. Add input dataset stats:
   - input_row_count
   - input_tf_count (0 for sequence-only inputs)
   - input_tfbs_count (for sequence inputs, treat as unique sequence count)
   - sampling_fraction (library size / input pool size; define precisely and document)

6. Add provenance for binding-site inputs:
   - For each placement, attach site_id and source (when provided) by propagating index-aligned
     provenance through sampling.

### Phase 2 - Reproducibility: library identity + solver provenance

7. Add:
   - sampling_library_index (1,2,3... within plan/source)
   - sampling_library_hash (sha256 of stable serialization of library entries + labels)
   - optional sampling_resample_index (if useful to distinguish new vs first build)

8. Add solver provenance:
   - dense_arrays_version
   - solver_status (if available)
   - solver_objective (if available)
   - solver_solve_time_s (measure around each yielded solution)

### Phase 3 - UX: run manifest (and small CLI upgrades)

9. Emit a run-level JSON manifest, e.g., outputs/meta/run_manifest.json, containing:
   - config hash, run_id, timestamp
   - per input + plan: generated, duplicates rejected, failed solutions, resamples, libraries_built,
     stall events
   - solver backend/strategy/options summary
   - optional histograms (compression_ratio, gc_total)

10. Add a CLI command like dense summarize (or extend workspace listing) to read and pretty-print the manifest.

### Phase 4 - Performance / resilience (optional but worthwhile)

11. Rewrite gap_fill.random_fill() to construct GC-valid strings directly instead of rejection
    sampling (keep semantics, remove unnecessary failure modes).

---

## Design decisions to fortify intent

### A) Lock down coordinate semantics

Document in the metadata schema (description text) that:

- used_tfbs_detail.offset is always in final sequence coordinates.
- offset_raw is solver/core coordinates (pre-gap-fill).
- end = offset + length (half-open interval) is best practice.

This prevents downstream confusion and makes coverage plots correct by default.

### B) Treat provenance as index-aligned data, not inferred strings

Do not encode site provenance into tf:tfbs strings. Keep them as fields:

- For each placement: site_id, source, and maybe library_index (the motif's index in the sampled
  library).

### C) Add one escape hatch for future metadata without schema churn

Consider adding a top-level field like extras: dict to the metadata schema (optional), so future
incremental info does not force schema edits. Keep high-value fields top-level; keep experimental
or rare fields in extras.

---

## Codex CLI prompt packs

These are written to be dropped into Codex as implement-this-patch tasks.

### Prompt Pack 1 - Fix placement offsets + enrich placement detail (correctness)

```text
You are editing the DenseGen repo.

Goal:
Fix a correctness bug: used_tfbs_detail.offset must reflect FINAL sequence coordinates after gap fill.
Also enrich each placement dict with derived fields: offset_raw, offset (final), length, end, and pad_left.

Constraints:
- Keep backward compatibility: used_tfbs_detail must still include keys tf/tfbs/orientation/offset.
- Do NOT change the top-level metadata keys yet unless necessary.
- If gap_fill_end == "5prime" and gap_fill_used is true, shift offsets by gap_fill_bases.
- Store original solver offsets as offset_raw.
- Add per placement: length = len(tfbs), end = offset + length.
- Also add pad_left = gap_fill_bases if 5prime else 0.

Scope:
- Update core/pipeline.py (_compute_used_tf_info and/or where used_tfbs_detail is finalized).
- Fix promoter motif detection to consider ALL promoter constraints (not just pcs[0]).

Acceptance:
- If gap_fill_end=5prime and gap length is N, then every placement offset increases by N and end increases by N.
- If gap_fill_end=3prime or gap_fill_used=false, offsets remain unchanged.
- Promoter motifs set must union upstream/downstream across all promoter_constraints.

Add minimal unit tests:
- Create a tiny dummy sol object with:
  - sequence = "AAAA" (len 4)
  - library = ["TT", "GG"]
  - offset_indices_in_order() returning [(0,0), (2,1)] for two placements
- Simulate gap_fill_used=true, gap_fill_end=5prime, gap_fill_bases=3 and assert offsets shift.
Use pytest, and keep tests independent of dense_arrays installation.
```

### Prompt Pack 2 - Add GC metrics (gc_total, gc_core) to metadata schema + derivation

```text
Goal:
Add gc_total and gc_core to per-record metadata.

Definitions:
- gc_core: GC fraction of sol.sequence (pre-gap-fill core)
- gc_total: GC fraction of final_seq (after padding)

Implementation:
- Add fields to core/metadata_schema.py:
  - gc_total: numbers.Real, required, not None
  - gc_core: numbers.Real, required, allow_none=False (or allow_none only if sol.sequence can be missing)
- Update core/metadata.py build_metadata() to accept gc_total/gc_core and include them.
- Update core/pipeline.py to compute these values per record.
  - Use uppercase counting consistent with gap_fill._gc_fraction, but do not import private helpers.
  - Implement a small local helper: gc_fraction(seq) -> float.

Acceptance:
- validate_metadata passes for newly built records.
- gc_total always exists and is between 0 and 1.
- gc_core always exists and is between 0 and 1.
- If no padding is used, gc_total == gc_core when sol.sequence == final_seq.
```

### Prompt Pack 3 - Add input dataset stats + sampling fraction

```text
Goal:
Add input dataset stats to metadata so users know the size of the input pool vs sampled library.

Add these top-level metadata fields (core/metadata_schema.py + build_metadata):
- input_row_count: int (required)
- input_tf_count: int (required)
- input_tfbs_count: int (required)
- sampling_fraction: numbers.Real (required, allow_none=True if division not meaningful)

Definitions:
- input_row_count: number of rows/sequences in the source pool before sampling (after source validation).
- input_tf_count:
  - for binding_sites: number of unique TFs in meta_df
  - for sequence_library/usr_sequences/pwm_sampled: 0
- input_tfbs_count:
  - for binding_sites: number of unique tfbs sequences in meta_df
  - for sequence_library/usr_sequences/pwm_sampled: number of unique sequences in the pool
- sampling_fraction:
  - if pool_strategy=="full": 1.0
  - else: len(sampled_library) / max(1, input_row_count) or / input_tfbs_count (choose ONE and document in metadata_schema description)

Implementation:
- Compute these stats in core/pipeline.py when loading the source and when building the sampled library.
- Plumb into build_metadata.

Acceptance:
- Works for binding_sites, sequence_library, usr_sequences, and pwm_* inputs.
- validate_metadata passes.
```

### Prompt Pack 4 - Binding-site provenance per placement (site_id, source)

```text
Goal:
When binding_sites input provides site_id and/or source columns, propagate those to placements.

Implementation strategy:
- Carry index-aligned arrays alongside library_for_opt and regulator_labels:
  - site_id_by_index: list[str|None]
  - source_by_index: list[str|None]
- For pool_strategy=full:
  - derive site_id_by_index/source_by_index directly from lib_df if those columns exist
- For subsampling:
  - Update core/sampler.py TFSampler.generate_binding_site_subsample() to ALSO return provenance lists aligned with returned sites/labels.
  - Keep backwards compatibility by returning a dict inside the existing info return (e.g. info["site_id_by_index"], info["source_by_index"]) OR change return signature and update call sites cleanly.
- Update _compute_used_tf_info to attach:
  - site_id and source to each used_tfbs_detail entry when available.

Acceptance:
- For binding_sites inputs with site_id/source configured, used_tfbs_detail entries include those keys for non-promoter placements.
- For other input types, those keys are absent or set to None (but do not break consumers).
- No changes to tfbs_parts formatting.
```

### Prompt Pack 5 - Library identity per resample (index + hash)

```text
Goal:
Allow grouping records by the exact sampled library.

Add metadata fields:
- sampling_library_index: int (required)
- sampling_library_hash: str (required)

Definition:
- sampling_library_index increments each time a new library is built for a given (source, plan) execution.
- sampling_library_hash is sha256 over a stable serialization of:
  - motifs in library_for_opt in order
  - regulator_labels aligned to motifs
  - (optional) site_id/source aligned if available
Use a stable delimiter and explicit "None" placeholders.

Implementation:
- core/pipeline.py: maintain a counter per _process_plan_for_source invocation.
- Compute hash each time _build_library returns.
- Add to sampling_info or pass directly to build_metadata; but final values must appear as top-level metadata keys.
- Update core/metadata_schema.py and build_metadata accordingly.

Acceptance:
- Two records produced from the same library in the same resample share the same hash and index.
- Different libraries (different sampling) have different hash.
```

### Prompt Pack 6 - Solver provenance: dense_arrays version + status + solve time

```text
Goal:
Record solver provenance and per-solution solve timing.

Add metadata fields:
- dense_arrays_version: str (required, allow_none=True if not discoverable)
- solver_status: str (required, allow_none=True)
- solver_objective: numbers.Real (required, allow_none=True)
- solver_solve_time_s: numbers.Real (required, allow_none=True)

Implementation:
- In adapters/optimizer/dense_arrays.py, wrap solution generation for all strategies to measure solve time:
  - measure time.monotonic() around each solver call / next(generator)
  - attach solve time to the solution object as an attribute like sol._densegen_solve_time_s
- In core/pipeline.py, read getattr(sol, "_densegen_solve_time_s", None)
- Discover dense_arrays version:
  - prefer getattr(dense_arrays, "__version__", None)
  - fallback: importlib.metadata.version("dense-arrays") or ("dense_arrays") depending on package name, handle exceptions
- solver_status/objective: attempt getattr(sol, "status", None), getattr(sol, "objective", None), getattr(sol, "objective_value", None) with conservative fallbacks.

Acceptance:
- Fields exist on all records (may be None if unavailable).
- For optimal strategy, solver_solve_time_s is non-null and >0 for non-trivial cases.
```

### Prompt Pack 7 - Run-level manifest (JSON) + CLI surface

```text
Goal:
Write a run-level JSON summary manifest so users do not have to scan per-record metadata.

Implementation:
- Create core/run_manifest.py (or similar) defining a RunManifest dataclass with:
  - run_id, created_at, config_sha256, run_root, solver backend/strategy/options/strands
  - per (input_name, plan_name): generated, duplicates_skipped, failed_solutions, total_resamples, libraries_built, stall_events
- In core/pipeline.py:
  - accumulate these stats during _process_plan_for_source
  - write manifest at the end of run_pipeline to outputs/meta/run_manifest.json
- In cli.py:
  - add a summarize command that loads and prints outputs/meta/run_manifest.json in a Rich table

Acceptance:
- Running a pipeline produces outputs/meta/run_manifest.json.
- summarize prints a readable table without needing parquet/usr outputs.
- Manifest writing is atomic (write temp then rename).
```

### Prompt Pack 8 - Make gap fill deterministic + GC-valid without rejection sampling

```text
Goal:
Replace rejection sampling in core/postprocess/gap_fill.py random_fill() with direct construction so it never fails when the GC window is feasible.

Algorithm:
- Compute lo = ceil(length * gc_min), hi = floor(length * gc_max).
- If lo > hi:
  - strict: raise
  - adaptive: relax to [0, length] and mark relaxed
- Choose gc_count uniformly between lo and hi.
- Construct a list of bases:
  - gc_count bases chosen from {G,C}
  - length-gc_count bases chosen from {A,T}
- Shuffle and join.
- attempts should be 1 (or 0 when length=0).
Return the same info dict keys as before.

Acceptance:
- strict mode never fails for feasible windows.
- adaptive mode matches previous behavior for infeasible windows but with deterministic construction.
- gc_actual always within final window.
```

---

## Implementation notes / gotchas

- validate_metadata rejects unknown top-level keys. Every new top-level field must be added to
  META_FIELDS or record writing will crash.
- Nested dicts/lists are permissive about extra keys, but top-level is not.
- If sinks flatten metadata keys into Parquet columns, update output adapters to include new fields.
- Update plotting only after new columns exist; prefer plots that tolerate missing columns or make
  them optional.

---

## Recommended prompt order

1. Prompt Pack 1 (offset correctness + promoter constraints)
2. Prompt Pack 2 (GC metrics)
3. Prompt Pack 5 (library index/hash)
4. Prompt Pack 6 (solver provenance + timing)
5. Prompt Pack 3 (input stats + sampling fraction)
6. Prompt Pack 4 (site_id/source provenance)
7. Prompt Pack 7 (run manifest + CLI summarize)
8. Prompt Pack 8 (gap fill optimization)

---

If you want, we can also produce a single mega prompt for Codex that implements Phases 0-2 in one
PR (offset fix + GC + library hash + solver timing), but in practice reviewability is better with
the prompt packs above.
