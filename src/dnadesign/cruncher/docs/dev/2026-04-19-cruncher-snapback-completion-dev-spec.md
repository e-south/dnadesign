## Cruncher Snapback Completion Dev Spec

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-19

> **Last updated by:** cruncher-maintainers on 2026-04-19
> **Historical note:** this developer spec captures the 2026-04-19 design cut. The shipped solve lane was later promoted to co-design-first `single_nick_snapback_solve_v3`. Use [`../guides/snapback_workflow.md`](../guides/snapback_workflow.md), [`../reference/snapback_artifacts.md`](../reference/snapback_artifacts.md), and [`../reference/architecture.md`](../reference/architecture.md) for the active workflow, ranking, and artifact layout.

### Contents
- [Purpose](#purpose)
- [Relationship To Prior Spec](#relationship-to-prior-spec)
- [Skill Composition Decision](#skill-composition-decision)
- [Constraints And Quality Bar](#constraints-and-quality-bar)
- [Problem Statement](#problem-statement)
- [Current-State Gap Summary](#current-state-gap-summary)
- [Boundary And Compatibility Decisions](#boundary-and-compatibility-decisions)
- [Alignment Target](#alignment-target)
- [Required Domain Semantics](#required-domain-semantics)
- [Public Workflow Contract](#public-workflow-contract)
- [Explicit V2 Spec Contract](#explicit-v2-spec-contract)
- [Historical Solve V2 Spec Contract](#historical-solve-v2-spec-contract)
- [Geometry And Pairing Rules](#geometry-and-pairing-rules)
- [Ranking Policy](#ranking-policy)
- [Artifact Contract](#artifact-contract)
- [Failure And Degraded-Mode Contract](#failure-and-degraded-mode-contract)
- [Implementation Plan](#implementation-plan)
- [Verification](#verification)
- [Rejected Alternatives](#rejected-alternatives)
- [Known Risks And Next Increment](#known-risks-and-next-increment)

## Purpose

Define the next implementation contract for `cruncher snapback` so the lane can honestly answer the narrow design question:

> among allowed nickases, orientations, nick boundaries, cap layouts, and short foldback homology layouts, what is the shortest viable single-nick snapback design under an explicit geometry contract?

This document completes the information architecture started by the slice-1 tracer bullet.

It is still intentionally narrow:

- single intended nick only
- sequence and geometry reasoning only
- no thermodynamic folding prediction
- no RT/processivity or in vivo scoring
- no hidden biological inference beyond the declared post-nick topology model

## Relationship To Prior Spec

This is a follow-on spec to:

- [2026-04-19-cruncher-snapback-dev-spec.md](2026-04-19-cruncher-snapback-dev-spec.md)

The earlier spec correctly defined the peer-lane architecture, shared `nickases` seam, explicit tracer-bullet workflow, and fail-fast artifact model.

This follow-on spec does **not** replace that work. It defines the missing semantics needed to move from:

- "validate this authored exact foldback candidate"

to:

- "model nick-relative snapback geometry and search for the shortest admissible candidate"

Implementation alignment rule:

- replace the tracer-bullet `v1` contract with a clean `v2` cut
- do not keep dual-contract runtime support inside `snapback`
- keep the old `v1` spec only as historical documentation

## Skill Composition Decision

- Primary skill: `pragmatic-programming-principles`
- Paired skill: `code-documentation`

Why:

- the main task is to codify explicit domain contracts and reversible boundary changes
- the requested output is an implementation-facing developer spec aligned to audited gaps

## Constraints And Quality Bar

- Keep `sample` unchanged.
- Keep `yiu` unchanged.
- Keep `cassette` unchanged apart from already-completed shared `nickases` reuse.
- Do not preserve the tracer-bullet `snapback v1` runtime contract.
- Add no hidden fallback to `cassette`, `yiu`, or heuristic rescue behavior.
- Fail fast on malformed geometry, ambiguous nick placement, incompatible orientation normalization, and bounded-search truncation without visibility.
- Keep the next increment reversible:
  - explicit `v2` contract with no hidden fallback
  - bounded solver before any wider biological modeling

## Problem Statement

The current lane validates one authored design composed as:

- `input.sequence`
- `hinge`
- `foldback_arm`

That is enough for a tracer bullet, but not enough for the real snapback question.

The actual design problem requires first-class reasoning about:

- where the duplexed pre-nick section ends
- where the intended nick boundary lands on the released strand
- how far from the left edge that boundary sits
- how much released sequence lies between the nick and the start of retained homology
- what the cap/turn length is
- how long the paired homology segment is
- whether one or two mismatches are allowed in that homology
- which orientation achieves the earliest viable nick while still nicking the normalized top strand

## Current-State Gap Summary

The completion work exists to close these audited gaps:

1. No `solve` workflow exists yet, so the lane cannot optimize.
2. The model has no first-class `nick_boundary`, `pre_nick_duplex_window`, or `released_prefix` semantics.
3. The current explicit model collapses:
   - cap length
   - homology length
   - ligatable terminal duplex length
   - full uninterrupted duplex length
4. Coordinate semantics are inconsistent:
   - spans are described as half-open
   - nick-window matching currently behaves inclusively
5. Orientation normalization exists only internally.
6. Preset-only catalog use is allowed by schema but blocked in workflow.

See:

- [2026-04-19-cruncher-snapback-dev-spec.md](2026-04-19-cruncher-snapback-dev-spec.md)
- [2026-04-19-retron-p4-hairpin-variant-audit.md](2026-04-19-retron-p4-hairpin-variant-audit.md)

## Boundary And Compatibility Decisions

### Clean-cut contract update

The command family remains:

- `cruncher snapback init-workspace|validate|design|solve|show`

Implementation stance:

- replace `single_nick_snapback_v1` with `single_nick_snapback_v2`
- add the historical `single_nick_snapback_solve_v2` design cut later superseded by `single_nick_snapback_solve_v3`
- `show` must distinguish explicit versus solve bundles from bundle metadata
- `init-workspace` should scaffold only `v2` examples

### Shared ownership remains unchanged

Allowed:

- `snapback` imports `nickases.*`

Disallowed:

- `snapback` imports `cassette` planner/solver internals
- `snapback` imports `yiu.*`
- `snapback` reuses YIU mismatch semantics directly

## Alignment Target

The minimal complete snapback model must be able to evaluate or search designs in this normalized frame:

1. A canonical top strand is defined left to right as the authored reference axis.
2. A duplex-only pre-nick window defines where nickase recognition may occur.
3. A single intended nick boundary is resolved on that top strand after orientation normalization.
4. The released top-strand segment to the right of that nick boundary is decomposed into:
   - released prefix
   - retained homology segment
   - cap/turn segment
   - foldback arm
5. The foldback arm pairs back onto the retained homology segment under an explicit mismatch policy.
6. Candidate ranking prefers:
   - earliest valid nick boundary from the left
   - then shortest added sequence
   - then smaller duplex burden

This is the smallest contract that aligns with the stated design objective without expanding into thermodynamics or retron biology claims.

## Required Domain Semantics

### Coordinate model

Use two distinct coordinate types:

- `CoordinateSpan`
  - half-open, zero-based, `[start, end)`
- `BoundaryPosition`
  - zero-based boundary between bases, valid in `[0, len(sequence)]`

Rule:

- spans and boundaries must never share a type

### Canonical strand and orientation model

Define a public orientation policy:

- `canonical_top_strand`
  - the reference top strand, always reported 5' to 3' left to right
- `normalize_to_top_strand_nick`
  - when true, solver may flip site orientation but only admits candidates whose resolved intended nick occurs on the canonical top strand
- `release_direction`
  - fixed to `left_to_right_from_nick` in `v2`

Reason:

- this matches the real operator mental model better than raw `primary/complement`
- it keeps internal nickase scanning reusable without leaking scanner-internal terms into the public contract

### Pre-nick duplex placement

Add:

- `pre_nick_duplex_window`
  - half-open window inside the authored top strand where the sequence is assumed duplex before nicking

All intended nickase recognition sites for snapback `v2` must lie fully inside this window.

### Intended nick boundary

Add:

- `nick_boundary_window`
  - allowed boundary interval for the intended nick
- `nick_boundary_from_left`
  - derived metric on the canonical top strand

For explicit `v2`:

- the selected intended boundary must be unique within `nick_boundary_window`

For solve `v2`:

- the solver optimizes `nick_boundary_from_left`

### Post-nick active-strand decomposition

For `v2`, the nick is the single snapback origin. After nicking, the released top strand is no longer the active structural strand; the active post-nick model is the remaining complement strand folding back toward the former nick position.

The explicit active-strand topology is:

- `retained_homology_window`
  - the canonical-top interval that begins exactly at `nick_boundary`
- derived `source_cap_window`
  - the canonical-top suffix from `retained_homology_window.end` to `input_sequence.end`
- `cap_sequence`
  - any authored cap extension appended after the source-side suffix
- derived `effective_cap_sequence = source_cap_sequence + cap_sequence`
  - the true unpaired turn between the retained stem and the foldback arm
- `foldback_arm`
  - sequence that pairs back onto `retained_homology_window`

Compatibility metrics remain published:

- `released_prefix_nt = 0`
- `retained_start_from_nick = 0`

The lane no longer permits a free nick-to-stem offset. `retained_homology_window.start` must equal `nick_boundary`.

### Cap semantics

Replace `hinge` in `v2` with:

- `cap_sequence`
- derived `cap_extension_nt = len(cap_sequence)`
- derived `cap_nt = len(effective_cap_sequence)`

Do not use `hinge` as the public name in `v2`.

Reason:

- "hinge" is too generic
- the current user intent is specifically a cap/turn region
- the effective loop must include any source-side suffix that remains between retained stem and foldback start

`v2` cap invariant:

- `cap_nt` is fixed at `3`
- `search.cap_nt` is therefore fixed at `{min: 3, max: 3}`
- the solver derives the authored cap-extension length from `3 - source_cap_nt`

### Homology policy

Add:

- `homology_policy.max_mismatches`
- `homology_policy.min_paired_bp`
- `homology_policy.max_paired_bp`

`v2` minimum support:

- `max_mismatches` in `{0, 1, 2}`
- default `min_paired_bp = 3` if omitted
- explicit `min_paired_bp` is preferred in authored specs when the design intent depends on a stricter floor

Derived outputs must include:

- `paired_bp`
- `mismatch_count`
- `mismatch_positions`

Protected-region rule:

- if `retained_homology_window` overlaps `protected_region`, mismatch positions inside that overlap are invalid
- the protected interval may still participate in perfect retained pairing
- `snapback` must not treat `protected_region` as passive metadata

### Ligation-adjacent duplex semantics

Separate:

- `terminal_ligatable_duplex_bp`
  - contiguous paired run adjacent to the modeled ligation junction
- `max_uninterrupted_duplex_bp`
  - longest contiguous paired run anywhere in the modeled post-nick state

Do not define both metrics as `len(foldback_arm)`.

### Extra nick-site policy

Extend the current extra-site logic to support:

- `forbid_additional_target_strand_nicks`
- `forbid_any_additional_nicks`
- report:
  - count
  - strand
  - boundary
  - site orientation

## Public Workflow Contract

### Commands

```bash
uv run cruncher snapback init-workspace WORKSPACE
uv run cruncher snapback validate --spec configs/snapback/<name>.snapback.yaml
uv run cruncher snapback design --spec configs/snapback/<name>.snapback.yaml
uv run cruncher snapback solve --spec configs/snapback/<name>.snapback.solve.yaml
uv run cruncher snapback show --run <run_dir>
```

### Version behavior

- `validate|design` accept only `single_nick_snapback_v2`
- historical design cut: `solve` emitted only `single_nick_snapback_solve_v2` reports and `single_nick_snapback_v2` hit bundles
- `show` distinguishes explicit versus solve bundles from bundle metadata without guessing

## Explicit V2 Spec Contract

File suffix:

```text
<workspace>/configs/snapback/<name>.snapback.yaml
```

Top-level contract:

```yaml
snapback:
  schema_version: 2
  contract: single_nick_snapback_v2
  name: teto_bpu10i_cap_v2

input:
  canonical_top_strand:
    sequence: <full authored top strand 5to3>
    protected_region: {start: <int>, end: <int>}
    pre_nick_duplex_window: {start: <int>, end: <int>}

design:
  nickase:
    variant_id: Nt.Bpu10I
    catalog:
      preset: neb_nicking_v1
      additional_paths: []
  orientation_policy:
    normalize_to_top_strand_nick: true
    release_direction: left_to_right_from_nick
  single_nick_goal:
    nick_boundary_window: {min: <int>, max: <int>}
  topology:
    retained_homology_window: {start: <int>, end: <int>}
    cap_sequence: <exact DNA>
    foldback_arm: <exact DNA>
    homology_policy:
      max_mismatches: 1
      min_paired_bp: 4
      max_paired_bp: 6
  constraints:
    max_added_nt: 10
    max_uninterrupted_duplex_bp: 8
    forbid_additional_target_strand_nicks: true
  sequence_quality:
    gc_fraction: {min: 0.0, max: 0.75}
    max_homopolymer_run: 4

output:
  run_dir: outputs/snapback
  emit_visual_contracts: true
```

### Explicit V2 parsing rules

- reject unknown keys
- reject ambiguous coordinate semantics
- reject `pre_nick_duplex_window` outside sequence bounds
- reject `retained_homology_window` outside sequence bounds
- reject `retained_homology_window.start < nick_boundary` after intended nick resolution
- reject `cap_sequence` containing non-ACGT
- reject `foldback_arm` containing non-ACGT
- reject `homology_policy.max_mismatches > 2` in `v2`
- reject `preset` plus `additional_paths` collisions that shadow the selected `variant_id` ambiguously

### Explicit V2 derived metrics

The report must publish:

- `nick_boundary`
- `nick_boundary_from_left`
- `released_prefix_nt`
- `retained_start_from_nick`
- `cap_nt`
- `cap_extension_nt`
- `paired_bp`
- `mismatch_count`
- `terminal_ligatable_duplex_bp`
- `max_uninterrupted_duplex_bp`
- `added_nt`
- `extra_nick_event_count`

## Historical Solve V2 Spec Contract

File suffix:

```text
<workspace>/configs/snapback/<name>.snapback.solve.yaml
```

Top-level contract:

```yaml
snapback_solve:
  schema_version: 2
  contract: single_nick_snapback_solve_v2

input:
  canonical_top_strand:
    sequence: <full authored top strand 5to3>
    protected_region: {start: <int>, end: <int>}
    pre_nick_duplex_window: {start: <int>, end: <int>}

catalog:
  preset: neb_nicking_v1
  additional_paths: []

nickase_policy:
  allowed_variant_ids: [Nt.Bpu10I, Nt.BbvCI]
  normalize_to_top_strand_nick: true

goal:
  nick_boundary_window: {min: 0, max: 8}
  retained_start_from_nick: {min: 0, max: 0}

search:
  retained_homology_length: {min: 4, max: 8}
  cap_nt: {min: 3, max: 3}
  max_added_nt: 12
  max_mismatches: 1
  max_enumerated_candidates: 50000
  max_search_nodes: 100000
  max_hits: 25
  materialize_top_k: 5

constraints:
  terminal_ligatable_duplex_bp: {min: 4, max: 6}
  max_uninterrupted_duplex_bp: 8
  forbid_additional_target_strand_nicks: true

sequence_quality:
  gc_fraction: {min: 0.0, max: 0.75}
  max_homopolymer_run: 4

output:
  run_dir: outputs/snapback_solves
  emit_visual_contracts: true
```

### Solve V2 generation rules

The solver must enumerate over:

- allowed nickase variants
- resolved site orientations
- intended nick boundaries inside the declared boundary window
- retained homology lengths
- derived source-side cap lengths from `input_sequence`
- concrete cap-extension sequences whose total effective cap stays at `3 nt`
- concrete foldback-arm sequences compatible with `max_mismatches`

The solver must not enumerate:

- outside `pre_nick_duplex_window`
- outside bounded search settings
- retained windows that do not start at the resolved nick
- candidates whose effective cap loop is not exactly `3 nt`
- candidates that mutate the protected region

Materialized hits must round-trip through `single_nick_snapback_v2`.

## Geometry And Pairing Rules

### Canonical post-nick model

For `v2`, the modeled post-nick active-strand sequence is:

```text
retained_homology + source_cap_sequence + cap_sequence + foldback_arm
```

Where:

- `retained_homology` is taken from the authored top strand and begins at the nick
- `source_cap_sequence` is the canonical-top suffix after `retained_homology_window.end`
- `cap_sequence` is the authored cap extension
- `foldback_arm` is authored or solver-generated

The released top strand may still be shown in the exposed QA view, but it is not the modeled foldback substrate.

### Pairing evaluation

The foldback arm pairs back onto the retained homology in reverse-complement orientation under:

- `max_mismatches`
- `min_paired_bp`
- `max_paired_bp`

The evaluator must emit an explicit alignment result rather than reducing everything to full-length perfect pairing.

Minimum alignment outputs:

- paired positions
- mismatch positions
- terminal ligation-adjacent run
- longest uninterrupted paired run

### Coordinate semantics

Rule:

- all spans use half-open zero-based coordinates
- all boundaries use integer boundary positions
- reports must publish both `coordinate_semantics` and `boundary_semantics`

No mixed interpretation is allowed.

## Ranking Policy

### Candidate admissibility

A candidate is admissible only if:

- exactly one intended nick resolves inside `nick_boundary_window`
- the intended nick is on the normalized top strand
- the intended recognition site lies fully inside `pre_nick_duplex_window`
- `retained_homology_window.start == nick_boundary`
- `retained_start_from_nick == 0`
- `cap_nt == 3`
- pairing satisfies `homology_policy`
- `terminal_ligatable_duplex_bp` is within requested bounds
- `max_uninterrupted_duplex_bp` is within requested bounds
- protected-region invariants hold
- extra nick-site policy is satisfied
- sequence-quality constraints are satisfied

### Primary ranking key

Rank admissible candidates by:

1. `nick_boundary_from_left`
2. `added_nt`
3. `max_uninterrupted_duplex_bp`
4. `extra_nick_event_count`
5. `gc_distance`
6. `homopolymer_penalty`
7. `cap_extension_nt`
8. lexical stability key

Why:

- the first biological objective is to nick as early as possible
- the nick is the fixed snapback origin, so there is no longer a second nick-to-stem distance objective
- only then does total added sequence length dominate

### Low-GC stance

Low GC remains a preference, not the primary objective.

Use:

- hard bounds in admissibility
- distance-from-preferred-range only as a late ranking key

Do not make GC the first or second ranking key.

## Artifact Contract

### Explicit run directory

```text
<workspace>/outputs/snapback/<spec.name>/<design_id>/
```

Required artifacts:

- `meta/snapback_manifest.json`
- `meta/snapback_status.json`
- `analysis/reports/report.json`
- `analysis/reports/report.md`
- `export/table__candidates.csv`
- `provenance/spec_used.yaml`
- `provenance/nickase_catalog.yaml`

Required new `v2` table/report fields:

- `nick_boundary`
- `nick_boundary_from_left`
- `released_prefix_nt`
- `retained_start_from_nick`
- `cap_nt`
- `paired_bp`
- `mismatch_count`
- `terminal_ligatable_duplex_bp`
- `max_uninterrupted_duplex_bp`

Optional visual surfaces:

- `views/pre_nick_duplex.v1.json`
- `views/post_nick_exposed.v1.json`
- `views/post_nick_foldback.v1.json`
- `views/pre_nick_duplex.snapback_visual.v1.json`
- `views/post_nick_exposed.snapback_visual.v1.json`
- `views/post_nick_foldback.snapback_visual.v1.json`
- `views/views_manifest.v1.json`

The producer-owned QA views remain the snapback-specific semantic source of truth.
The `*.snapback_visual.v1.json` files are shared `snapback_visual_v1` publication contracts for downstream BaseRender rendering.
Optional sibling `baserender_jobs/*.job.yaml` files may reference those evidence-map contracts by path.

They must show coordinate truth and topology decomposition only:

- `pre_nick_duplex`
  - canonical top strand plus complement row in duplex context
  - one shared `Nick / origin` boundary
  - intended site, protected region, retained homology, source-cap/effective-cap, and foldback-arm spans
- `post_nick_exposed`
  - same coordinate axis after nicking
  - released top span plus the active complement strand
  - the nick is still the single snapback origin boundary
  - retained homology, source-cap/effective-cap, and foldback-arm spans are interpreted on the active strand
- `post_nick_foldback`
  - post-nick local active-strand foldback sequence with origin at coordinate `0`
  - pair map, mismatch positions, terminal ligatable run, max uninterrupted duplex, and protected-overlap projection

The visual contracts may stay on version `1` if they remain structurally compatible, but they must include the `v2` semantic fields and remain independent of baserender output geometry.

### Solve run directory

```text
<workspace>/outputs/snapback_solves/<solve_id>/
```

Required artifacts:

- `solve_report.json`
- `solve_report.md`
- `table__hits.csv`
- `solve_manifest.json`
- `solve_status.json`
- `specs/input_solve_spec.yaml`
- `specs/resolved_catalog.yaml`

If the solver uses a preset-only catalog:

- still emit `resolved_catalog.yaml`
- do not require an original local overlay path

### Show contract

`snapback show` must fail fast on:

- missing manifest
- missing report
- missing resolved catalog snapshot
- disagreement between manifest and status
- declared visual artifacts missing
- unsupported contract version

## Failure And Degraded-Mode Contract

### No silent fallback

The lane must never:

- fall back to `cassette`
- fall back to `yiu`
- reinterpret `v1` as `v2`
- silently widen nick-boundary windows
- silently drop mismatch limits
- silently treat preset catalogs as local catalogs

### Explicit statuses

Use explicit statuses:

- `satisfied`
- `unsatisfied`
- `no_hits`
- `invalid_spec`
- `invalid_catalog`
- `search_truncated`

For explicit mode:

- `satisfied`
- `unsatisfied`
- `invalid_spec`
- `invalid_catalog`

`validate` and `design` must both surface typed `invalid_catalog` reports for:

- missing preset entries
- missing local catalog files
- malformed catalog payloads
- unknown `variant_id` inside an otherwise resolved catalog

For solve mode:

- all of the above plus `no_hits` and `search_truncated`

### Degraded mode policy

Allowed degraded behavior:

- bounded search truncation only

Required visibility:

- machine-visible in solve status
- operator-visible in CLI output
- contract-visible in solve report metadata

### Error semantics

Errors and issues must include:

- stable code
- short actionable message
- relevant field/path context
- boundary or window values when geometry is involved
- candidate-count or truncation metadata when search is involved

## Implementation Plan

### Slice 2A: contract and metric refactor

Goal:

- add `v2` explicit models as the only live `snapback` contract

Required changes:

- add `BoundaryPosition` and boundary-range models
- add explicit `pre_nick_duplex_window`
- add `cap_sequence`
- add `homology_policy`
- add explicit report metrics for nick-relative geometry
- allow preset-only catalogs by always snapshotting the resolved merged catalog

Stop condition:

- `validate|design|show` can round-trip a `v2` explicit spec

### Slice 2B: geometry evaluator

Goal:

- replace the current exact full-length reverse-complement check with a proper pairing evaluator

Required changes:

- compute released prefix
- compute mismatch positions
- compute `terminal_ligatable_duplex_bp`
- compute `max_uninterrupted_duplex_bp`
- publish version-aware views

Stop condition:

- explicit `v2` can represent zero-prefix, early-boundary, short-cap cases and mismatch-tolerant homology

### Slice 3: bounded solver

Goal:

- implement `snapback solve`

Required changes:

- solver models
- bounded enumeration
- deterministic ranking
- solve artifacting
- hit materialization into explicit `v2`

Stop condition:

- solver can compare multiple nickases/orientations and produce top-ranked explicit bundles

### Slice 4: scaffold and docs alignment

Goal:

- make the default operator experience match `v2`

Required changes:

- update `init-workspace`
- ship a `v2` example with one preset-only and one local-overlay example
- update reference docs and help text

## Verification

Required tests:

- new `v2` load tests
- new `v2` planner tests
- reverse-orientation site tests
- normalized top-strand nick tests
- boundary `0` tests
- preset-only catalog tests
- retained-start-from-nick ordering tests
- mismatch-tolerant homology tests
- coordinate-semantics tests
- solve boundedness/truncation tests
- CLI help and artifact round-trip tests for `solve`

Required checks:

- `uv run ruff check .`
- `uv run ruff format --check .`
- targeted `pytest` for new `snapback` and `nickases` coverage
- `uv run python -m dnadesign.devtools.docs_checks --repo-root .`

## Rejected Alternatives

### Keep dual `v1` and `v2` runtime support

Rejected because:

- it would silently change an already-shipped contract
- it would make old bundles ambiguous

### Reuse `hinge` as the public `v2` cap field

Rejected because:

- it hides the real cap/turn semantics
- it preserves the current ambiguity the audit called out

### Overload YIU mismatch semantics for snapback homology

Rejected because:

- YIU is a 4 nt ligation-window mismatch engine
- snapback needs a topology-aware foldback homology policy

### Optimize by total added sequence before earliest nick boundary

Rejected because:

- it does not match the stated design objective
- it can prefer a later nick with slightly fewer added nucleotides, which is not the intended behavior

## Known Risks And Next Increment

Known risks:

- the exact ligation-adjacent duplex metric may need one more naming pass during implementation
- mismatch support beyond `2` may require a wider search budget than the intended narrow lane
- some nickases may produce ambiguous early boundaries when orientation-normalized to top-strand nicking

Next smallest increment:

- implement `v2` explicit models and pairing evaluator first
- do not start solve mode until `v2` explicit can express:
  - boundary `0`
  - zero-length released prefix
  - explicit cap length separate from homology
  - one-mismatch homology
  - preset-only catalog provenance
