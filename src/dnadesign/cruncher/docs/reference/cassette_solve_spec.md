## Cassette solve spec reference

**Owner:** dnadesign-maintainers
**Doc kind:** reference
**Audience:** cassette workflow users and maintainers
**Updated by:** cruncher-maintainers on 2026-03-24
**Applies to:** `configs/cassettes/*.cassette.solve.yaml`
**Last verified:** 2026-03-24
**Primary artifacts:** `solve_report.json`, `table__hits.csv`, materialized explicit hit bundles

### Contents
- [File location](#file-location)
- [Canonical shape](#canonical-shape)
- [Field semantics](#field-semantics)
- [Blacklist semantics](#blacklist-semantics)
- [Search guardrails](#search-guardrails)

### File location

Cassette solve specs must live at:

```text
<workspace>/configs/cassettes/<name>.cassette.solve.yaml
```

The loader does not overload explicit `.cassette.yaml` specs with solve semantics.

### Canonical shape

```yaml
cassette_solve:
  schema_version: 1
  topology:
    stem5p_arm_pattern: NNNNNNNNNNNN
    loop_pattern: TTT
  construct_context:
    left_flank: ""
    right_flank: ""
    evaluation_scope: cassette_plus_flanks
  nick_goal:
    target_strand: primary
    left_nick_window: {start: 2, end: 6}
    right_nick_window: {start: 20, end: 24}
    bounded_segment_length: {min: 10, max: 18}
  assignment_policy:
    allowed_left_variant_ids: [Nb.BbvCI, Nt.AlwI]
    allowed_right_variant_ids: [Nt.BbvCI, Nt.AlwI]
    forbidden_intended_variant_ids: [Nt.CviPII]
    forbidden_intended_specificity_ids: []
    allow_same_variant: true
    allow_same_specificity_opposite_variant: true
  site_blacklist:
    forbidden_any_site_specificity_ids: [BsmAI]
    forbidden_unintended_site_specificity_ids: [BspQI]
    forbidden_any_site_variant_ids: []
    scope: evaluation_context
  sequence_blacklist:
    forbidden_literals: [GAATTC]
    forbidden_iupac_motifs: [CGTCTC]
    forbid_reverse_complements: true
    scope: evaluation_context
  sequence_quality:
    gc_fraction: {min: 0.35, max: 0.65}
    max_homopolymer_run: 4
  catalog:
    preset: neb_nicking_v1
    additional_paths: []
  search:
    max_hits: 25
    max_enumerated_candidates: 100000
    max_search_nodes: 250000
    min_pairwise_hamming_distance: 2
    bounded_segment_target: 14
    gc_target: 0.5
    materialize_top_k: 5
  output:
    run_dir: outputs/cassette_solves
    write_render_contract: true
```

### Field semantics

- `schema_version`: must be `1`.
- `topology.stem5p_arm_pattern`: IUPAC pattern for the free 5' stem arm.
- `topology.loop_pattern`: IUPAC pattern for the loop. Length is fixed by the pattern.
- `nick_goal.target_strand`: required solve-mode target strand. There is no default.
- `nick_goal.left_nick_window`, `nick_goal.right_nick_window`: inclusive bond-boundary windows.
- `assignment_policy.*`: allowed and forbidden intended-assignment choices for left/right nickase variants.
- `site_blacklist.*`: recognition-site occurrence rules in `cassette_only` or `evaluation_context`.
- `sequence_blacklist.*`: literal or IUPAC sequence motifs excluded independently of catalog identities.
- `catalog.preset`: built-in preset ID such as `neb_nicking_v1`.
- `catalog.additional_paths`: local overlay catalogs appended after the preset. Duplicate IDs fail fast.
- `search.max_hits`: number of ranked hits returned after diversity filtering.
- `search.materialize_top_k`: number of top hits written as explicit per-hit bundles.

### Blacklist semantics

- `forbidden_intended_variant_ids` and `forbidden_intended_specificity_ids` only restrict intended left/right assignments.
- `forbidden_any_site_specificity_ids` bans recognition-site instances anywhere in scope, including intended hits.
- `forbidden_unintended_site_specificity_ids` bans extra occurrences outside the chosen intended left/right hits.
- `forbidden_any_site_variant_ids` bans occurrences of specific catalog variants when variant-specific scanning is needed.
- `forbidden_literals` and `forbidden_iupac_motifs` operate on sequence text, not catalog identity.

Site-occurrence blacklists are specificity-first because sequence occurrence is a motif question, not just a catalytic-variant question.

### Search guardrails

- `topology.stem5p_arm_pattern` is capped at 64 nt and `topology.loop_pattern` at 32 nt in the current first-phase solver.
- `assignment_policy.allowed_left_variant_ids x allowed_right_variant_ids` must stay at or below 256 distinct pairings, and duplicate IDs are rejected at load time.
- `max_enumerated_candidates` is the cap on fully concrete candidates sent through the explicit planner.
- `max_search_nodes` is the cap on DFS search nodes explored before the tree is exhausted.
- `max_hits` is capped at 128 and `materialize_top_k` at 32 so result retention and artifact fan-out stay bounded.
- `max_enumerated_candidates` is capped at 250000 and `max_search_nodes` at 500000; larger values are rejected instead of being silently clamped.
- When either cap is hit, the solve report emits warnings instead of silently pretending the search was exhaustive.
- The solver also keeps a bounded internal hit buffer so abundant valid hits do not cause unbounded memory growth.
- Solve mode does not expose inert toggles for energetic hairpin validation or multi-intended nick modes.
