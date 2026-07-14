## Snapback workflow

**Owner:** dnadesign-maintainers
**Doc kind:** guide
**Audience:** snapback workflow users and maintainers
**Last updated by:** cruncher-maintainers on 2026-07-13
**Applies to:** `uv run cruncher snapback init-workspace|validate|design|solve|target-search|show`
**Last verified:** 2026-07-13
**Primary artifacts:** `analysis/reports/report.json`, `analysis/reports/solve_report.json`, `analysis/views/views_manifest.v1.json`, `analysis/materialized_hits/hit_<rank>/`

### Contents
- [Why this exists](#why-this-exists)
- [Current workflow scope](#current-workflow-scope)
- [Core terms](#core-terms)
- [Workspace layout](#workspace-layout)
- [Minimal specs](#minimal-specs)
- [Standard command sequence](#standard-command-sequence)
- [Current solve ranking](#current-solve-ranking)
- [Outputs](#outputs)
- [Failure modes](#failure-modes)
- [When not to use snapback](#when-not-to-use-snapback)

### Why this exists

`snapback` is a narrow Cruncher lane for one question:

> given one authored top strand and an allowed nickase catalog, can Cruncher validate one single-nick foldback design or search a bounded space of single-nick foldback candidates under an explicit geometry contract?

It is separate from `sample`, `cassette`, `scar_nick`, and `yiu`.

Use `snapback` when you need:

- deterministic validation of one authored single-nick foldback design
- bounded search over nick boundary, retained homology length, cap extension, motif-compatible site edits, and foldback-arm choices
- target-first catalog search for an exact preserved-site geometry with the shortest feasible authored top strand
- explicit reports, stable tables, and a three-state QA triptych for the accepted design or top-ranked hits
- released-product evaluation of a retained post-release object without forcing the nickase site itself into the final 6 nt budget

### Current workflow scope

Current scope:

- `snapback validate` checks one explicit `single_nick_snapback_v2` spec
- `snapback design` writes one explicit bundle under a stable workspace output root
- `snapback solve` runs bounded co-design search under `single_nick_snapback_solve_v3`
- `snapback target-search` asks the catalog whether an exact preserved-site geometry exists and reports the nearest later-boundary fallbacks
- `snapback released-design` validates one explicit two-stage precursor under `single_nick_released_snapback_v1`
- `snapback released-target-search` searches paired nickase plus release-enzyme combinations on the exposed post-release bottom-strand geometry
- `snapback released-show` inspects released-product bundles and fails fast on projection and manifest drift
- `snapback show` inspects explicit or solve bundles and fails fast on manifest, visual, or materialized-hit drift

Current non-scope:

- no thermodynamic folding prediction
- no RT/processivity, ligation-yield, or in vivo scoring
- no retron, bulge-topology, or protein-binding interpretation layer
- no fallback to `sample`, `cassette`, `scar_nick`, or `yiu`
- no multi-nick or excision workflow

The lane is geometry-first. It answers whether a design satisfies the declared single-nick foldback contract and how accepted candidates rank under the current deterministic policy.

For the two-stage precursor sibling lane, use [`snapback_released_workflow.md`](snapback_released_workflow.md).

### Core terms

- **explicit design**: one authored `single_nick_snapback_v2` spec at `configs/snapback/<name>.snapback.yaml`
- **solve spec**: one bounded search spec at `configs/snapback/<name>.snapback.solve.yaml`
- **authored top strand**: the reference sequence and coordinate frame for the snapback lane
- **nick boundary**: the resolved zero-based closed boundary where the intended nick lands
- **retained homology**: the nick-anchored segment that remains paired to the foldback arm after nicking
- **source cap sequence**: the sequence already present between the retained homology and the end of the authored top strand
- **cap sequence**: the authored cap extension appended after the authored top strand
- **effective cap sequence**: `source_cap_sequence + cap_sequence`; in the live contract this must total exactly `3 nt`
- **foldback arm**: the appended sequence that pairs against retained homology in the post-nick foldback state
- **QA triptych**: the three published snapback states: `pre_nick_duplex`, `post_nick_exposed`, and `post_nick_foldback`

The preferred public vocabulary is `canonical_top_strand`, `nick_boundary`, `retained_homology`, `source_cap`, `cap extension`, and `foldback_arm`. Treat the code identifier literally and avoid repeating it as prose when authored top strand is clearer. Treat older `v1` or historical `solve_v2` names as historical only.

### Workspace layout

Store snapback specs and nickase catalogs inside the workspace:

```text
<workspace>/
  configs/
    runbook.yaml
    snapback/
      demo.snapback.yaml
      demo.snapback.solve.yaml
  inputs/
    nickases/
      local.nickases.yaml
  outputs/
    design/
    solve/
```

Create a preserved-site workspace with `uv run cruncher snapback init-workspace snapback_lab`
when you want the scaffolded explicit and solve examples described here.

### Minimal specs

Minimal explicit design:

```yaml
snapback:
  schema_version: 2
  contract: single_nick_snapback_v2
  name: demo_snapback
input:
  canonical_top_strand:
    sequence: ATGCAAAT
    protected_region: {start: 0, end: 8}
    pre_nick_duplex_window: {start: 0, end: 8}
design:
  nickase:
    variant_id: Nt.Early
    catalog:
      additional_paths: [inputs/nickases/local.nickases.yaml]
  single_nick_goal:
    nick_boundary_window: {min: 1, max: 1}
  topology:
    retained_homology_window: {start: 1, end: 5}
    cap_sequence: ""
    foldback_arm: TGCA
    homology_policy:
      max_mismatches: 0
      min_paired_bp: 4
      max_paired_bp: 4
  constraints:
    terminal_ligatable_duplex_bp: {min: 4, max: 4}
    max_uninterrupted_duplex_bp: 4
    max_added_nt: 5
output:
  run_dir: outputs/design
```

Minimal solve:

```yaml
snapback_solve:
  schema_version: 3
  contract: single_nick_snapback_solve_v3
  name: demo_snapback_solve
input:
  canonical_top_strand:
    sequence: AAAAAAAT
    protected_region: {start: 0, end: 8}
    pre_nick_duplex_window: {start: 0, end: 8}
catalog:
  preset: neb_nicking_v1
  additional_presets: [thermo_nicking_v1]
  additional_paths: []
search:
  min_paired_bp: 3
  max_added_nt: 5
  max_mismatches: 0
  max_enumerated_candidates: 4096
  max_search_nodes: 4096
  max_hits: 4
  materialize_top_k: 2
output:
  run_dir: outputs/solve
```

The scaffolded examples written by `cruncher snapback init-workspace` use the
same spec shapes shown here.

### Standard command sequence

```bash
set -euo pipefail
cd src/dnadesign/cruncher/workspaces/snapback_lab
cruncher() { uv run cruncher "$@"; }

# Validate one explicit design.
cruncher snapback validate --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml

# Materialize one explicit bundle.
cruncher snapback design --spec configs/snapback/demo_teto_bpu10i_cap.snapback.yaml --force-overwrite

# Inspect the explicit bundle and drift checks.
cruncher snapback show --run outputs/design

# Run bounded solve search and materialize the top hits.
cruncher snapback solve --spec configs/snapback/demo_teto_catalog_scan.snapback.solve.yaml --force-overwrite

# Ask the catalog for an exact origin-junction hit with stem 3 and cap 3.
cruncher snapback target-search --json

# Inspect the solve bundle and materialized explicit hit bundles.
cruncher snapback show --run outputs/solve
```

Notes:

- `validate` exits nonzero when the explicit spec is unsatisfied.
- `design` still writes a run directory for an unsatisfied explicit spec so the issue report is preserved.
- `solve` returns `satisfied`, `no_hits`, or `search_truncated` from the live search path.
- `target-search` does not write a bundle; it prints a typed exact-hit / near-hit report to stdout.
- the scaffold keeps `Nt.Bpu10I` as the explicit authored example, but the solve demo searches the broader NEB + Thermo preset catalog
- `show` reads snapback-specific bundles only and refuses drift instead of guessing.
- the released-product lane is a sibling contract surface and does not overload preserved-site `target-search` or `solve`; use [`snapback_released_workflow.md`](snapback_released_workflow.md) and the checked-in `de033` workspace for dual-enzyme examples

### Target-first catalog search

Use `snapback target-search` when the question is:

- can the current nickase catalog realize an exact `nick_boundary / paired_bp / cap_nt` geometry?
- which entry and orientation achieve that with the recognition site preserved exactly?
- what is the nearest later-boundary fallback when the exact boundary is impossible?

Use `snapback solve` when the question is:

- given one authored top strand, what bounded co-design candidates satisfy the live snapback contract?

The two lanes are intentionally different. `target-search` treats authored top-strand length as a search output. `solve` treats the authored top strand as authored input.

### Current solve ranking

The live solve lane uses deterministic lexicographic ranking over accepted candidates. Today the order is:

1. `nick_boundary_from_left`
2. `paired_bp`
3. `cap_extension_nt`
4. `added_nt`
5. `extra_nick_event_count`
6. `site_mutation_count`
7. nickase catalog priority
8. `gc_fraction_added`
9. `max_homopolymer_run_added`
10. `max_uninterrupted_duplex_bp`
11. remaining lexical tie-breakers

This is the current implementation order. Treat it as authoritative for the live lane.

### Outputs

Snapback writes stable workspace roots:

- explicit design bundle under `<workspace>/outputs/design/`
- solve bundle under `<workspace>/outputs/solve/`
- materialized explicit top-hit bundles under `<workspace>/outputs/solve/analysis/materialized_hits/hit_<rank>/`
- the top-level solve bundle is summary-only; `analysis/views/`, `baserender_jobs/`, and `plots/` belong to materialized explicit hit bundles, not `outputs/solve/`

Each satisfied explicit bundle can publish:

- one machine report and one markdown report
- one candidate table
- three producer-owned QA views
- three shared `snapback_visual_v1` contracts plus one JSONL triptych
- one `views_manifest.v1.json`
- one optional `baserender_jobs/snapback_triptych.job.yaml`

See [`../reference/snapback_artifacts.md`](../reference/snapback_artifacts.md) for the full file-by-file layout.

### Failure modes

Common fail-fast cases:

- the spec path is outside `configs/snapback/`
- the nickase catalog path escapes the workspace or fails to load
- no intended nick lands inside the requested boundary window and duplex window
- the intended nick resolves only on the non-top strand when top-strand normalization is required
- `retained_homology_window.start` does not equal the resolved nick boundary
- `effective_cap_sequence` is not exactly `3 nt`
- protected-region mismatches appear inside retained homology
- terminal ligatable duplex or uninterrupted duplex limits are exceeded
- extra nick policies reject the candidate
- `show` detects manifest/status drift, visual drift, or materialized-hit drift

`search_truncated` is not silent degradation. It means search hit a configured bound and the report preserves that fact explicitly.

### When not to use snapback

Do not use `snapback` when the real question is:

- thermodynamic fold stability
- retron processivity or RT-DNA yield
- ligation chemistry or wet-lab yield prediction
- motif binding competence after structure edits
- bulge, bubble, or scaffold-topology design beyond the fixed snapback geometry contract

Those are separate problems. `snapback` stays narrow on purpose.
