## Cassette solve workflow

**Owner:** dnadesign-maintainers
**Doc kind:** guide
**Audience:** cassette workflow users and maintainers
**Updated by:** cruncher-maintainers on 2026-03-25
**Applies to:** `uv run cruncher cassette solve`
**Last verified:** 2026-03-25
**Primary artifacts:** `solve_report.json`, `table__hits.csv`, `baserender_hits_contract.json`, per-hit explicit cassette bundles

### Contents
- [Why solve exists](#why-solve-exists)
- [Current solve scope](#current-solve-scope)
- [Bootstrap an isolated workspace](#bootstrap-an-isolated-workspace)
- [Minimal solve spec](#minimal-solve-spec)
- [Standard command sequence](#standard-command-sequence)
- [Outputs](#outputs)
- [Guardrails](#guardrails)

### Why solve exists

`cassette validate|design` assumes you already know the concrete cassette sequence and intended nickase assignments.
`cassette solve` adds a separate search layer that:

- chooses concrete `stem5p_arm` and `loop` bases from IUPAC patterns
- enumerates allowed intended left/right nickase assignments
- validates every concrete candidate through the existing explicit cassette planner
- returns deterministic ranked hits and optionally materializes the top `k`

### Current solve scope

Current solve behavior:

- fixed stem length from `topology.stem5p_arm_pattern`
- fixed loop length from `topology.loop_pattern`
- built-in preset loading via `catalog.preset`, including `neb_nicking_v1`
- additive local overlay catalogs via `catalog.additional_paths`
- specificity-level site blacklists and literal/IUPAC sequence blacklists
- deterministic accepted-pool admission, policy-driven selection, and top-k materialization

Current non-scope:

- variable-length stem or loop search
- remote catalog sync
- energetic hairpin validation
- direct rendering beyond the per-hit render contract handoff

### Bootstrap an isolated workspace

If you do not already have a cassette workspace root, generate one explicitly:

```bash
uv run cruncher cassette init-workspace --output /tmp/cassette_lab
cd /tmp/cassette_lab
```

The scaffold writes three pressure-tested solve specs under `configs/cassettes/`:

- `demo_hairpin_fast.cassette.solve.yaml`
- `demo_hairpin_balanced.cassette.solve.yaml`
- `demo_hairpin_deep_mmr.cassette.solve.yaml`

This scaffold is intentionally cassette-specific:

- it creates only the directories cassette workflows need
- it does not auto-register as a general sampling workspace
- it refuses to overwrite a non-empty unowned root, so it will not trample sibling workspaces by accident
- it rejects symlinked output roots and symlinked ancestor directories, so the scaffold lands exactly where you asked for it
- `cassette_workspace_manifest.json` records the fast, balanced, and deep MMR profile budgets so you can compare them without reopening each YAML

### Minimal solve spec

```yaml
cassette_solve:
  schema_version: 1
  topology:
    stem5p_arm_pattern: NNNNNCCTCAGC
    loop_pattern: TTT
  construct_context:
    left_flank: ""
    right_flank: ""
    evaluation_scope: cassette_plus_flanks
  nick_goal:
    target_strand: primary
    left_nick_window: {start: 0, end: 0}
    right_nick_window: {start: 24, end: 24}
    bounded_segment_length: {min: 24, max: 24}
  assignment_policy:
    allowed_left_variant_ids: [Nt.BbvCI]
    allowed_right_variant_ids: [Nb.BbvCI]
    forbidden_intended_variant_ids: []
    forbidden_intended_specificity_ids: []
    allow_same_variant: true
    allow_same_specificity_opposite_variant: true
  site_blacklist:
    forbidden_any_site_specificity_ids: []
    forbidden_unintended_site_specificity_ids: []
    forbidden_any_site_variant_ids: []
    scope: evaluation_context
  sequence_blacklist:
    forbidden_literals: []
    forbidden_iupac_motifs: []
    forbid_reverse_complements: true
    scope: evaluation_context
  sequence_quality:
    gc_fraction: {min: 0.35, max: 0.65}
    max_homopolymer_run: 4
  catalog:
    preset: neb_nicking_v1
    additional_paths: []
  search:
    max_hits: 10
    max_enumerated_candidates: 10000
    max_search_nodes: 250000
    bounded_segment_target: 24
    gc_target: 0.5
    materialize_top_k: 3
    selection:
      policy: greedy_hamming
      pool_size: 64
      distance_metric: hamming
      min_pairwise_distance: 2
  output:
    run_dir: outputs/cassette_solves
    write_render_contract: true
```

### Selection policies

Solve selection happens after the explicit planner accepts concrete candidates. The bounded accepted pool is then reduced by one of three deterministic policies:

- `score_only`: best score tiers only, no diversity pressure.
- `greedy_hamming`: compatibility behavior. Candidates stay score-sorted and must satisfy the configured minimum Hamming distance.
- `mmr`: opt-in score/diversity tradeoff over the bounded accepted pool. Better score tiers stay preferred, but candidates inside the same tier compete on sequence diversity.

Results remain honest about boundedness:

- `ACCEPTED_POOL_TRUNCATED` means valid candidates were dropped because the accepted pool filled.
- `SELECTION_RESULTS_POOL_BOUNDED` means returned hits are best only among the retained pool.
- `SELECTION_RESULTS_SEARCH_BOUNDED` means search budgets stopped exploration before the full search space was exhausted.
- `SELECTION_POLICY_LIMITED_HITS` means the selection policy itself returned fewer hits than the accepted pool could otherwise support, typically because diversity constraints filtered the pool.

### Standard command sequence

```bash
set -euo pipefail

# 0) Optional: bootstrap an isolated cassette workspace first.
uv run cruncher cassette init-workspace --output /tmp/cassette_lab
cd /tmp/cassette_lab

# 1) Search and print a human summary.
uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin_balanced.cassette.solve.yaml

# 2) Search and capture machine-readable JSON.
uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin_balanced.cassette.solve.yaml --json
```

Policy examples:

```bash
# Shorter bounded run.
uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin_fast.cassette.solve.yaml

# Deeper opt-in MMR over the bounded accepted pool.
uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin_deep_mmr.cassette.solve.yaml
```

The returned hit sets can differ even when the same valid candidates are discovered because `score_only` and `greedy_hamming` follow score-first order, while `mmr` intentionally spreads hits across score-equivalent sequence neighborhoods.

### Outputs

Solve runs are written under:

```text
<workspace>/outputs/cassette_solves/<solve_id>/
```

Primary artifacts:

- `solve_report.json`
- `solve_report.md`
- `table__hits.csv`
- `baserender_hits_contract.json` when enabled
- `solve_manifest.json`
- `solve_status.json`
- `specs/input_solve_spec.yaml`
- `specs/resolved_catalog.yaml`
- `hits/<rank>_<hit_id>/resolved_candidate.cassette.yaml`
- `hits/<rank>_<hit_id>/report.json`
- `hits/<rank>_<hit_id>/report.md`
- `hits/<rank>_<hit_id>/manifest.json`
- `hits/<rank>_<hit_id>/status.json`
- `hits/<rank>_<hit_id>/render_contract.json` when enabled

`solve_report.json` now includes `selection_summary`, and `solve_status.json` mirrors lightweight selection telemetry so automation can tell whether a hit set was search-bounded, pool-bounded, or default-policy selected.
`baserender_hits_contract.json` is a data-only `generic_features` bundle for solve hits, so downstream visualization can consume the run without scraping every per-hit directory or importing baserender internals from Cruncher.

Each materialized hit round-trips through the explicit cassette lane.
Preflight `invalid_spec` and `invalid_catalog` results still persist the top-level solve bundle when the workspace can be derived, but they stop before writing `specs/resolved_catalog.yaml` or any per-hit artifacts.

### Guardrails

`cassette solve` is intentionally bounded:

- `topology.stem5p_arm_pattern` and `topology.loop_pattern` are capped to first-phase safety limits of 64 nt and 32 nt
- duplicate assignment IDs are rejected, and the left/right intended assignment cross-product must stay at or below 256 pairings
- `search.max_enumerated_candidates` caps full concrete candidates sent to the explicit planner
- `search.max_search_nodes` caps DFS search nodes before the search tree is exhausted
- `search.selection.pool_size` bounds the accepted pool retained for final selection
- `search.max_hits` is capped at 128 and `search.materialize_top_k` at 32 so retained hits and per-hit bundles stay bounded
- oversized search budgets are rejected at spec load time instead of being silently clamped
- the solve report distinguishes search truncation from accepted-pool truncation instead of collapsing both into one warning
- impossible assignment/window geometry fails early as `invalid_spec`
- plain CLI output surfaces warnings instead of silently degrading
- `solve_status.json` preserves warnings, warning codes, selection policy, `search_truncated`, accepted-pool truncation state, and policy-underfill telemetry without requiring the full report
