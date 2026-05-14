## Cruncher Snapback Dev Spec

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-19

> **Last updated by:** cruncher-maintainers on 2026-04-19

### Contents
- [Purpose](#purpose)
- [Skill Composition Decision](#skill-composition-decision)
- [Constraints And Quality Bar](#constraints-and-quality-bar)
- [Problem Statement](#problem-statement)
- [Boundary And Ownership Decisions](#boundary-and-ownership-decisions)
- [Public Workflow Contract](#public-workflow-contract)
- [Domain Model](#domain-model)
- [Explicit And Solve Spec Contracts](#explicit-and-solve-spec-contracts)
- [Ranking And Search Policy](#ranking-and-search-policy)
- [Artifact Contract](#artifact-contract)
- [Failure And Degraded-Mode Contract](#failure-and-degraded-mode-contract)
- [Implementation Plan](#implementation-plan)
- [Verification](#verification)
- [Rejected Alternatives](#rejected-alternatives)
- [Known Risks And Next Increment](#known-risks-and-next-increment)

## Purpose

Define the first implementation contract for a new peer Cruncher workflow family, `snapback`, that designs or searches for a short post-nick foldback cap.

This spec is implementation-facing. It is not a biology claim that Cruncher can predict RT processivity, ligation yield, or in vivo function. The v1 lane is limited to explicit geometry and sequence-contract reasoning.

## Skill Composition Decision

- Primary skill: `pragmatic-programming-principles`
- Paired skill: `code-documentation`

Why:

- the main task is architecture and delivery design under maintainability and fail-fast constraints
- the requested output is a durable developer artifact rather than only chat guidance

## Constraints And Quality Bar

- Keep `sample` unchanged. Snapback is not a sampling submode.
- Keep `yiu` unchanged. YIU remains mismatch-centric and fixed to a 4 nt internal junction window.
- Reuse nickase catalog semantics and cut normalization already proven in cassette workflows, but do not couple the new lane to cassette planner internals.
- Keep v1 honest:
  - topology and geometry only
  - no thermodynamic folding claims
  - no RT/processivity scoring
  - no silent biological inference from ligation or hairpin geometry
- Fail fast on malformed specs, unsupported states, missing catalog entries, and empty candidate sets.
- Keep the first delivery slice reversible and test-backed.

## Problem Statement

The v1 product question is:

> Given a protected payload boundary, a nickase rule, a designated nicked strand, and a bounded design space for a foldback cap, can Cruncher find the shortest added sequence that yields a geometrically ligatable post-nick foldback while keeping uninterrupted duplex burden below a declared cap?

This is narrower than cassette and different from YIU:

- cassette models a symmetric hairpin with exactly two intended nicks and no bulges or mismatches
- YIU models mismatch choice inside a fixed 4 nt ligation window
- snapback needs one intended nick and a post-nick foldback topology that can be shorter and more asymmetric than cassette allows

Biological motivation from the local retron audit should inform priorities but not broaden the software claim:

- avoid simple continuous stem extension
- preserve a compact protected motif scaffold
- prefer short uninterrupted duplex segments
- treat topology relief as a first-class design variable rather than overloading mismatch coordinates

See:

- [Cruncher architecture](../reference/architecture.md)
- [Cassette solve workflow](../guides/cassette_solve_workflow.md)
- [YIU workflow](../guides/yiu_workflow.md)
- [2026-04-19 retron P4 hairpin audit](2026-04-19-retron-p4-hairpin-variant-audit.md)

## Boundary And Ownership Decisions

### New peer workflow family

Add a new peer lane:

- `cruncher snapback init-workspace|validate|design|solve|show`

Do not overload:

- `cruncher sample`
- `cruncher yiu`
- `cruncher cassette`

This follows the existing Cruncher architecture rule that peer workflow families keep separate contracts, output trees, and orchestration seams.

### Package ownership

Create a new owner-local package boundary:

```text
src/dnadesign/cruncher/src/snapback/
```

Expected module shape:

```text
src/dnadesign/cruncher/src/snapback/
  __init__.py
  models.py
  load.py
  planner.py
  solve_models.py
  solver.py
  artifacts.py
  view_contracts.py
  errors.py
```

Application and CLI seams:

```text
src/dnadesign/cruncher/src/app/snapback_workflow.py
src/dnadesign/cruncher/src/app/snapback_solve_workflow.py
src/dnadesign/cruncher/src/app/snapback_workspace_service.py
src/dnadesign/cruncher/src/cli/commands/snapback.py
```

### Shared nickase seam

Before `snapback` consumes nickase catalogs, extract the reusable nickase catalog logic from cassette into a neutral shared seam:

```text
src/dnadesign/cruncher/src/nickases/
  __init__.py
  models.py
  catalog.py
```

This shared seam owns only:

- normalized nickase entry models
- raw cut notation parsing
- preset loading
- local overlay merge rules
- catalog export helpers

It does not own:

- cassette planning
- cassette artifact layout
- snapback scoring
- snapback topology rules

Pragmatic reason:

- this removes duplicated knowledge
- it prevents `snapback` from importing `cassette.*` domain modules
- it keeps the new lane reversible without widening cassette scope

### No hidden cross-family coupling

Allowed:

- `snapback` imports `nickases.*`
- `cassette` imports `nickases.*`

Disallowed:

- `snapback` imports `cassette.models`, `cassette.solver`, `cassette.artifacts`, or `cassette.view_contracts`
- `snapback` imports `yiu.*`

## Public Workflow Contract

### Workspace layout

```text
<workspace>/
  configs/
    snapback/
      <name>.snapback.yaml
      <name>.snapback.solve.yaml
  inputs/
    nickases/
      *.yaml
  outputs/
    snapback/
    snapback_solves/
```

### Command surface

Explicit lane:

```bash
uv run cruncher snapback init-workspace WORKSPACE
uv run cruncher snapback validate --spec configs/snapback/<name>.snapback.yaml
uv run cruncher snapback design --spec configs/snapback/<name>.snapback.yaml
uv run cruncher snapback show --run outputs/snapback/<name>/<design_id>
```

Solve lane:

```bash
uv run cruncher snapback solve --spec configs/snapback/<name>.snapback.solve.yaml
```

### Delivery stance

The first tracer bullet does not need full solve mode on day one.

Required order:

1. shared `nickases` seam extraction
2. explicit `snapback validate|design|show`
3. bounded `snapback solve`

This keeps the first vertical slice small enough to validate the information architecture before committing to search behavior.

## Domain Model

The v1 lane introduces only the following first-class primitives:

- `protected_region`
  - payload interval that snapback must not mutate
- `nickase_rule`
  - normalized motif plus cut-offset rule loaded from the shared nickase catalog
- `single_nick_goal`
  - exactly one intended nick on a declared strand and within a declared boundary window
- `foldback_arm`
  - designed sequence that can pair with a retained target segment after nicking
- `hinge`
  - designed spacer between retained sequence and foldback arm; may be unpaired or weakly constrained
- `ligatable_duplex_budget`
  - allowed terminal paired-length interval for the post-nick foldback junction
- `max_uninterrupted_duplex_bp`
  - hard cap on the longest contiguous complementary duplex run in the post-nick foldback state

### Invariants

- v1 models exactly one intended nick
- the protected region is immutable unless a future contract explicitly opts out
- the reported topology is the post-nick foldback state, not a pre-nick thermodynamic fold prediction
- all scored candidates are concrete `A/C/G/T` sequences
- geometry checks are deterministic and sequence-derived
- the winning candidate is reproducible from spec bytes plus catalog bytes

### Non-goals

V1 does not model:

- ViennaRNA Package energetics, including either the `RNA` Python module or
  the `RNAfold` CLI program
- RT pausing or RNase H processing
- TetR binding affinity
- bulge energetics
- stochastic ligation probability
- multi-nick or excision workflows

## Explicit And Solve Spec Contracts

### Explicit spec

File suffix:

```text
<workspace>/configs/snapback/<name>.snapback.yaml
```

Top-level contract:

```yaml
snapback:
  schema_version: 1
  contract: single_nick_snapback_v1
  name: teto_bpu10i_cap

input:
  payload_context:
    sequence: <full input sequence>
    protected_region: {start: <int>, end: <int>}

design:
  nickase:
    variant_id: Nt.Bpu10I
    catalog:
      additional_paths: [inputs/nickases/local.nickases.yaml]
  single_nick_goal:
    target_strand: primary
    nick_window: {start: <int>, end: <int>}
  topology:
    foldback_arm: <exact DNA>
    hinge: <exact DNA>
    retained_target_window: {start: <int>, end: <int>}
  constraints:
    ligatable_duplex_budget: {min: <int>, max: <int>}
    max_uninterrupted_duplex_bp: <int>
    max_added_nt: <int>
  sequence_quality:
    gc_fraction: {min: <float>, max: <float>}
    max_homopolymer_run: <int>

output:
  run_dir: outputs/snapback
  emit_visual_contracts: true
```

### Solve spec

File suffix:

```text
<workspace>/configs/snapback/<name>.snapback.solve.yaml
```

Top-level contract:

```yaml
snapback_solve:
  schema_version: 1
  contract: single_nick_snapback_solve_v1

input:
  payload_context:
    sequence: <full input sequence>
    protected_region: {start: <int>, end: <int>}

catalog:
  preset: null
  additional_paths: [inputs/nickases/local.nickases.yaml]

nickase_policy:
  allowed_variant_ids: [Nt.Bpu10I, Nt.BbvCI]

goal:
  target_strand: primary
  nick_window: {start: <int>, end: <int>}
  retained_target_window: {start: <int>, end: <int>}

search:
  foldback_arm_pattern: NNNNNN
  hinge_pattern: NNN
  max_added_nt: 12
  max_enumerated_candidates: 50000
  max_search_nodes: 100000
  max_hits: 25
  materialize_top_k: 5

constraints:
  ligatable_duplex_budget: {min: 4, max: 8}
  max_uninterrupted_duplex_bp: 19

sequence_quality:
  gc_fraction: {min: 0.25, max: 0.75}
  max_homopolymer_run: 4

output:
  run_dir: outputs/snapback_solves
  emit_visual_contracts: true
```

### Parsing rules

- parse at the CLI/config boundary only
- reject unknown keys
- reject ambiguous IUPAC input in explicit mode
- reject missing catalog sources
- reject unsafe paths and `..` traversal
- reject `max_uninterrupted_duplex_bp < ligatable_duplex_budget.min`
- reject protected-region windows outside sequence bounds
- reject retained-target windows that overlap immutable protected bases if the topology would require mutating them
- reject solve specs that define neither preset nor local overlay catalogs

## Ranking And Search Policy

### Candidate admissibility

A candidate is admissible only if all of the following are true:

- exactly one intended nickase placement satisfies the single-nick goal
- the post-nick foldback arm pairs against the declared retained target window
- the terminal paired run adjacent to the ligation junction is within `ligatable_duplex_budget`
- the longest contiguous complementary run in the modeled post-nick state is `<= max_uninterrupted_duplex_bp`
- the protected region remains unchanged
- no forbidden extra nickase-site policy is violated
- sequence-quality constraints are satisfied

### Primary ranking key

Rank admissible candidates by:

1. `added_nt`
2. `uninterrupted_duplex_bp`
3. `extra_nickase_site_count`
4. `gc_distance`
5. `homopolymer_penalty`
6. lexical stability key

Rationale:

- this preserves the biological priority to minimize added structure first
- it keeps scoring deterministic and explainable
- it copies the style of cassette solve scoring without reusing cassette-specific semantics

### Search boundedness

Solve mode must expose boundedness explicitly:

- `max_enumerated_candidates`
- `max_search_nodes`
- `max_hits`
- `materialize_top_k`

If any bound truncates the search, emit warnings in report metadata instead of silently presenting results as exhaustive.

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

Optional visual surfaces:

- `views/pre_nick_linear.v1.json`
- `views/post_nick_foldback.v1.json`
- `views/views_manifest.v1.json`

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

Materialized hit bundles must round-trip through the explicit `single_nick_snapback_v1` contract.

### Show contract

`snapback show` must be read-only and fail fast on bundle drift:

- missing manifest
- missing report
- missing catalog snapshot
- disagreement between manifest and status
- missing declared visual contracts

## Failure And Degraded-Mode Contract

### No silent fallback

The lane must never:

- fall back to `cassette`
- fall back to `yiu`
- silently disable geometry constraints
- silently widen search bounds
- silently drop the protected-region constraint

### Explicit statuses

Use explicit statuses:

- `satisfied`
- `unsatisfied`
- `no_hits`
- `invalid_spec`
- `invalid_catalog`

### Degraded mode policy

There is no hidden degraded mode in v1.

The only allowed degraded behavior is search boundedness, and it must be:

- contract-visible in the solve report
- operator-visible in CLI output
- machine-visible in `solve_status.json`

If optional render or view publication is disabled by config, that is not degraded mode. It is an explicit authored mode.

### Error semantics

Errors must include:

- stable issue code
- short actionable message
- relevant field or path context
- enough candidate-count metadata to guide the next relaxation step when applicable

Recovery advice belongs in app/CLI reporting, not in pure domain models.

## Implementation Plan

### Slice 0: preparatory refactor

Goal:

- extract shared nickase catalog logic from cassette to `nickases`

Required evidence:

- cassette still passes unchanged behavior tests
- no `snapback` module exists yet

### Slice 1: explicit tracer bullet

Goal:

- deliver `snapback validate|design|show` for exact authored candidates

Scope:

- one explicit contract
- one deterministic planner
- report and artifact publication
- no solve mode yet

Why first:

- it validates the domain vocabulary
- it proves artifact layout and CLI seams
- it prevents premature search abstraction

### Slice 2: bounded solve

Goal:

- add `snapback solve` over bounded foldback-arm and hinge design space

Scope:

- deterministic enumeration
- bounded search telemetry
- top-k materialization

### Slice 3: visual hardening

Goal:

- stabilize pre-nick and post-nick visual contracts for operator inspection

Scope:

- keep visuals file-based and optional
- no direct renderer invocation from Cruncher

## Verification

### Required tests

Nickase seam:

- catalog normalization parity tests for current cassette fixtures
- preset plus overlay merge tests
- duplicate ID and malformed cut-notation failures

Snapback explicit lane:

- spec-load contract tests
- protected-region invariant tests
- single-intended-nick validation tests
- uninterrupted-duplex cap tests
- report/artifact publication tests
- CLI smoke tests for `validate`, `design`, and `show`

Snapback solve lane:

- bounded-search warning tests
- ranking determinism tests
- top-k materialization tests
- no-hit and invalid-spec failure tests

Regression:

- existing cassette tests stay green
- existing YIU tests stay green
- docs checks pass

### Commands

Targeted:

```bash
uv run pytest -q src/dnadesign/cruncher/tests/cassette
uv run pytest -q src/dnadesign/cruncher/tests/cli/test_cassette_cli.py
uv run pytest -q src/dnadesign/cruncher/tests/snapback
uv run pytest -q src/dnadesign/cruncher/tests/cli/test_snapback_cli.py
```

Repo docs:

```bash
uv run python -m dnadesign.devtools.docs.checks --repo-root .
```

### Current validation evidence

After adding this spec document:

- `uv run python -m dnadesign.devtools.docs.checks --repo-root .`

should pass before implementation begins, so the design artifact is checked in under the same docs contract as the rest of Cruncher.

## Rejected Alternatives

### 1. Add snapback as a YIU mode

Rejected because:

- YIU is explicitly mismatch-centric
- YIU rejects bulge and topology keys
- overloading mismatch coordinates with foldback topology would create a dishonest contract

### 2. Add snapback as a `sample` feature

Rejected because:

- `sample` is a broad fixed-length optimizer
- this problem is sequence-topology and nickase-contract specific
- it would mix unrelated state, artifacts, and docs routes

### 3. Extend cassette in place

Rejected for v1 because:

- cassette currently assumes a symmetric stem-loop plus exactly two intended nicks
- single-nick foldback semantics would force exceptions into a model that currently forbids them

### 4. Import cassette internals directly from snapback

Rejected because:

- it would couple two peer workflow families
- it would make later independent change harder
- the only real shared knowledge is the nickase catalog seam

## Known Risks And Next Increment

### Known risks

- the correct `ligatable_duplex_budget` thresholds are engineering placeholders until calibrated against lab results
- a local `Nt.Bpu10I` catalog overlay may be required until a broader preset is intentionally added
- topology-only scoring can rank a candidate that is geometrically plausible but biologically poor
- visual contracts may need one iteration after explicit runs reveal which state representation operators actually need

### Next increment

Implement Slice 0 and Slice 1 only:

1. extract `nickases`
2. add explicit `snapback` spec, planner, artifacts, and CLI
3. prove one authored `Nt.Bpu10I` example can validate and publish deterministically

Do not start solve mode until the explicit tracer bullet is stable and the report format is good enough for a human to reject or accept a candidate without reading raw JSON.
