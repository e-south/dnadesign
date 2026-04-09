## Cassette solve workflow

**Owner:** dnadesign-maintainers
**Doc kind:** guide
**Audience:** cassette workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-04-05
**Applies to:** `uv run cruncher cassette solve`
**Last verified:** 2026-03-25
**Primary artifacts:** `solve_report.json`, `table__hits.csv`, `views/top_hits.linear_duplex.v1.jsonl`, `views/top_hits.ssdna_hairpin.v1.jsonl`, and per-hit explicit cassette bundles

### Contents
- [Why solve exists](#why-solve-exists)
- [Current solve scope](#current-solve-scope)
- [Bootstrap a Runbook-Only Workspace](#bootstrap-a-runbook-only-workspace)
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
- direct renderer invocation from Cruncher; use the emitted `baserender_jobs/*.job.yaml` files instead

### Bootstrap a Runbook-Only Workspace

If you do not already have a cassette workspace root, generate one explicitly:

```bash
uv run cruncher cassette init-workspace cassette_lab
uv run cruncher workspaces list --root src/dnadesign/cruncher/workspaces
cd src/dnadesign/cruncher/workspaces/cassette_lab
```

The scaffold writes three pressure-tested solve specs under `configs/cassettes/`:

- `demo_hairpin_fast.cassette.solve.yaml`
- `demo_hairpin_balanced.cassette.solve.yaml`
- `demo_hairpin_deep_mmr.cassette.solve.yaml`

This scaffold is intentionally cassette-specific:

- it creates only the directories cassette workflows need
- it ships `configs/runbook.yaml`, so `workspaces list` reports it as `runbook-only`
- it still omits a generic `configs/config.yaml` because cassette flows do not use the sampling schema
- it refuses to overwrite a non-empty unowned root, so it will not trample sibling workspaces by accident
- it rejects symlinked output roots and symlinked ancestor directories, so the scaffold lands exactly where you asked for it
- `cassette_workspace_manifest.json` records the fast, balanced, and deep MMR profile budgets so you can compare them without reopening each YAML

If you want the same scaffold under a different parent, pass `--root /path/to/workspaces` or `--output /explicit/workspace/path`.

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
    emit_visual_contracts: true
    emit_baserender_jobs: true
    baserender_profiles: [duplex_qa, hairpin_qa, top_hits_duplex_qa, top_hits_hairpin_qa]
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

If you want the shortest scaffolded tutorial that starts from an empty root and ends with rendered PDFs, use
[`../demos/demo_cassette_workspace.md`](../demos/demo_cassette_workspace.md) first and return here for the full solve contract.

### Standard command sequence

```bash
set -euo pipefail

# 0) Optional: bootstrap a runbook-only cassette workspace first.
uv run cruncher cassette init-workspace cassette_lab
cd src/dnadesign/cruncher/workspaces/cassette_lab

# 1) Search and print a human summary.
uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin_balanced.cassette.solve.yaml

# 2) Search and capture machine-readable JSON.
uv run cruncher cassette solve --spec configs/cassettes/demo_hairpin_balanced.cassette.solve.yaml --json

# 3) Optional: validate or render the emitted jobs in place with baserender.
uv run baserender job validate outputs/cassette_solves/<solve_id>/baserender_jobs/top_hits_duplex.job.yaml
uv run baserender job run outputs/cassette_solves/<solve_id>/baserender_jobs/top_hits_duplex.job.yaml
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
- `solve_manifest.json`
- `solve_status.json`
- `specs/input_solve_spec.yaml`
- `specs/resolved_catalog.yaml`
- `views/top_hits.linear_duplex.v1.jsonl` when `output.emit_visual_contracts: true`
- `views/top_hits.ssdna_hairpin.v1.jsonl` when `output.emit_visual_contracts: true`
- `baserender_jobs/top_hits_duplex.job.yaml` when `output.emit_baserender_jobs: true`
- `baserender_jobs/top_hits_hairpin.job.yaml` when `output.emit_baserender_jobs: true`
- `renders/top_hits_duplex_qa_sheet.pdf` after you run the solve-level duplex job with `baserender`
- `renders/top_hits_hairpin_qa_sheet.pdf` after you run the solve-level hairpin job with `baserender`
- `output.emit_baserender_jobs` requires `output.emit_visual_contracts: true`
- `hits/hit_<rank>_<solution_id>/explicit/resolved_candidate.cassette.yaml`
- `hits/hit_<rank>_<solution_id>/explicit/report.json`
- `hits/hit_<rank>_<solution_id>/explicit/report.md`
- `hits/hit_<rank>_<solution_id>/explicit/manifest.json`
- `hits/hit_<rank>_<solution_id>/explicit/status.json`
- `hits/hit_<rank>_<solution_id>/views/linear_duplex.v1.json` when enabled
- `hits/hit_<rank>_<solution_id>/views/ssdna_hairpin.v1.json` when enabled
- `hits/hit_<rank>_<solution_id>/views/views_manifest.v1.json` when enabled
- `hits/hit_<rank>_<solution_id>/baserender_jobs/linear_duplex.job.yaml` when enabled
- `hits/hit_<rank>_<solution_id>/baserender_jobs/ssdna_hairpin.job.yaml` when enabled
- `hits/hit_<rank>_<solution_id>/renders/linear_duplex.pdf` after you run the emitted duplex job with `baserender`
- `hits/hit_<rank>_<solution_id>/renders/ssdna_hairpin.pdf` after you run the emitted hairpin job with `baserender`

`solve_report.json` now includes `selection_summary`, and `solve_status.json` mirrors lightweight selection telemetry plus the top-hit JSONL and job paths so automation can tell whether a hit set was search-bounded, pool-bounded, or default-policy selected without reopening every per-hit bundle.

The solve bundle is intentionally self-contained. The emitted jobs consume sibling `views/` contracts and write rendered PDFs to sibling `renders/` directories, so you do not need a separate baserender workspace to inspect results.

That workspace-scoped flow is the stable operator path:

- Cruncher writes solve-level or per-hit `views/`.
- Cruncher writes sibling `baserender_jobs/` that point at those local view files only.
- BaseRender writes rendered PDFs back into sibling `renders/` directories under the same solve bundle.

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
