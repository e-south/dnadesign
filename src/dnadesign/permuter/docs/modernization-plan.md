# Permuter Modernization Plan

**Type:** living dev spec
**Owner:** dnadesign-maintainers
**Owner-boundary:** `src/dnadesign/permuter`
**Status:** proposed
**Last verified:** 2026-05-24

## Plan Intent Summary

Modernize Permuter from a CLI/job-first mini-project into a contract-backed tool
with a stable public API, workspace-scoped execution, and schema-aware dataset
handling. The target outcome is a maintainable path from input biological
sequence to explicit variant records, model scores, plots, and downstream
selection artifacts without hidden coupling to package-local generated results.

This plan follows top-level `dnadesign` principles:

- parse inputs at boundaries, then operate on trusted internal structures
- fail fast on missing or invalid prerequisites
- keep tool boundaries explicit and avoid cross-tool internal imports
- accumulate repeated-run state under workspace roots, not ad hoc repository
  paths
- keep documentation layered, owner-local, and backed by executable checks
- use evidence-backed quality gates instead of convention-only contracts

## Worth-Doing Preflight

Best case: Permuter becomes the public variant-generation and in silico DMS
surface for DNA, codon, protein, and multi-site workflows, including future
reverse transcriptase scans from a provided protein or coding sequence.

Why it matters: the current package works, but its contracts have drifted across
legacy metric columns, newer observed/expected columns, RT-specific selection
logic, and package-local generated outputs. Hardening those seams now reduces
future maintenance cost and makes Permuter safe to call from studies, notebooks,
cluster jobs, and other `dnadesign` tools.

## Current Audit Baseline

Verified on 2026-05-24:

- `uv run permuter --help` passed.
- `uv run pytest -q src/dnadesign/permuter/tests` passed with 56 tests.
- All 9 checked-in job YAML files validated with `JobConfig`.
- No tracked Permuter code changes were present at audit time.

Primary findings:

- `evaluate` canonicalizes scores to `permuter__observed__*`, while
  `combine_aa` still consumes single-mutant scores from
  `permuter__metric__*`.
- Plot dispatch currently requires observed, expected, and `epistasis` columns
  for all plot types, which overfits multi-site selection datasets.
- Strict validation rejects unnamespaced derived columns such as `epistasis`,
  even though multi-site selection requires that column today.
- Jobs point at ignored package-local `results/` paths, so downstream
  reproducibility depends on local generated residue.
- The public CLI exists, but no stable `dnadesign.permuter.api` surface exists
  for programmatic use.

## Explicit Scope

In scope:

- dataset column contract modernization
- job/config parsing and schema validation hardening
- workspace model introduction
- public API facade for generation and evaluation planning
- protein-native or codon-policy-backed DMS request design
- docs and tests that make the contracts executable
- compatibility adapters for existing jobs and generated datasets

Out of scope for the first modernization pass:

- replacing Evo2 evaluator internals
- changing USR core semantics
- moving cross-tool schemas into shared contracts before a real cross-tool
  consumer needs them
- rewriting all protocols at once
- preserving deprecated identifiers as silent aliases
- committing generated `results/` artifacts

## Target Architecture

### Boundary Model

Permuter should have four explicit layers:

1. `api`
   Public, typed library entrypoints. No console output, no implicit filesystem
   writes unless the caller passes a workspace or output policy.

2. `contracts`
   Pydantic/dataclass request and result objects, dataset schema helpers,
   metric-column selectors, and compatibility adapters.

3. `runtime`
   Execution orchestration for generation, evaluation, plotting, validation,
   journaling, and workspace materialization.

4. `cli`
   Typer command bindings that parse user input and delegate to `api` or
   `runtime` services.

Existing protocol/evaluator implementations can remain under `src/` during the
transition, but new cross-module calls should target public contracts and
services rather than CLI functions.

### Workspace Model

Add a tool-local workspace convention:

```text
src/dnadesign/permuter/workspaces/<workspace_id>/
  README.md
  config.yaml
  inputs/
  jobs/
  outputs/
    datasets/<run_id>/records.parquet
    artifacts/<run_id>/
    logs/ops/
      audit/latest.json
      runbooks/
```

Workspace principles:

- `config.yaml` is the entry artifact for workspace-backed execution.
- `outputs/` is generated and ignored unless explicitly promoted.
- `jobs/` may hold migrated job YAMLs during compatibility.
- Downstream dependencies between DMS, combine, and selection runs are explicit
  graph edges in config, not hard-coded relative paths into `results/`.
- Existing `--job` behavior remains available during migration, but workspace
  commands become the preferred operational surface.

### Dataset Column Contract

Canonical columns:

- USR core: `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`,
  `created_at`
- variant identity: `permuter__var_id`
- generation metadata: `permuter__job`, `permuter__ref`,
  `permuter__protocol`, `permuter__round`, `permuter__modifications`
- observed metrics: `permuter__observed__<metric_id>`
- expected metrics: `permuter__expected__<metric_id>`
- interaction metrics: `permuter__interaction__epistasis__<metric_id>`
- vector outputs: `permuter__observed__<metric_id>` may be Arrow list columns
  when the evaluator declares a vector shape

Compatibility rule:

- Do not accept legacy `permuter__metric__<metric_id>` as runtime input.
- Re-materialize old datasets through `permuter evaluate` instead of carrying a
  read shim.
- Plot and protocol code must use the metric contract helpers rather than
  hand-building column names.

### Public API Direction

Initial public API objects:

- `ProteinDmsRequest`
- `CodingDnaDmsRequest`
- `NucleotideDmsRequest`
- `EvaluatorPlan`
- `PlotPlan`
- `PermuterWorkspace`
- `VariantRecord`
- `PermuterResult`

Initial API functions:

- `generate_variants(request) -> PermuterResult`
- `evaluate_variants(result_or_dataset, plan) -> PermuterResult`
- `materialize_result(result, workspace_or_output) -> DatasetRef`
- `validate_dataset(dataset, contract) -> ValidationReport`

Protein DMS policy:

- Protein-native enumeration should emit protein variant records without
  pretending a nucleotide sequence exists.
- If DNA or Evo2-DNA scoring is requested from a protein request, the caller must
  provide a coding DNA reference or an explicit codon design policy.
- Back-translation must be explicit and recorded in provenance.

## Ordered Modernization Checklist

### Slice 0: Lock Contracts Before Refactors

- Add a short contract glossary to the Permuter docs for observed, expected,
  interaction, metric selector, workspace, run, dataset, and artifact.
- Add tests that reproduce the current drift:
  - fresh `run -> evaluate -> combine_aa` should use the canonical metric
    selector
  - ordinary DMS plots should not require expected/epistasis columns
  - strict validation should accept the intended multi-site contract
- Decide and document the rejection rule for legacy `permuter__metric__*`.

Done when:

- the failing contract tests are written first
- each failing test names the expected replacement behavior
- no runtime refactor has started before the contract is explicit

### Slice 1: Metric and Dataset Schema Hardening

- Implement one metric selector module that resolves:
  - observed scalar metrics
  - observed vector metrics
  - expected metrics
  - interaction metrics
- Update `combine_aa`, plot modules, inspect, and validate to use the selector.
- Split dataset validation by dataset kind:
  - `dms_single`
  - `hairpin_scan`
  - `combine_aa`
  - `multisite_select`
  - `generic_permuter`
- Reject legacy metric columns with actionable errors.
- Update docs to remove stale `RECORDS.md` and `permuter__metric__*` claims.

Done when:

- package tests pass
- a temp-output `run -> evaluate -> validate -> plot` smoke test passes
- RT DMS result can feed `combine_aa` without depending on stale metric columns

### Slice 2: Read-Only Command Semantics

- Add pure validation/inspection paths that do not mutate `RECORD.md`.
- Keep CLI journaling opt-in or explicit for commands that are conceptually
  read-only.
- Make command help state whether a command mutates the dataset journal.

Done when:

- `validate` and `inspect` can run without changing file mtimes in pure mode
- tests assert pure mode does not append `RECORD.md`

### Slice 3: Workspace Shell

- Add workspace config models, loader, resolver, and validator.
- Add CLI commands:
  - `permuter workspace list`
  - `permuter workspace validate`
  - `permuter workspace inspect`
- Add one migrated RT workspace that models:
  - single-codon DMS
  - combine from DMS
  - multi-site selection
  - explicit dataset edges
- Keep existing jobs as compatibility presets.
- Implement `PERMUTER_OUTPUT_ROOT` or remove it from docs and help. Hidden
  fallback behavior is not allowed.
- Make invalid `output.layout` values fail fast instead of falling through to
  nested layout.

Done when:

- workspace validation fails fast on missing inputs, duplicate run ids, invalid
  dependencies, and invalid layouts
- workspace output paths stay under workspace `outputs/` unless an explicit
  external root is configured
- existing job CLI still works

### Slice 4: Public API Facade

- Add `dnadesign.permuter.api` with typed request/result objects.
- Refactor CLI commands to call API/runtime services rather than owning behavior.
- Keep filesystem materialization separate from pure variant generation.
- Add protein-native DMS enumeration.
- Add coding-DNA-backed protein DMS with explicit codon policy requirements.

Done when:

- a user can call `generate_variants(ProteinDmsRequest(...))` without touching
  the filesystem
- a user can materialize the same result into a workspace
- CLI and API share validation contracts

### Slice 5: Operability and Quality Integration

- Add status or inventory metadata only if Permuter gains repeated operational
  use that needs observation-plane discovery.
- Add docs-check coverage for the modernization spec links.
- Add package-data entries for workspace templates only after they become
  packaged defaults.
- Add coverage baseline updates after the new contract tests land.

Done when:

- quality evidence links point to real tests or docs checks
- workspace examples are packaged or clearly marked as source-only
- no generated result artifacts are required for tests

## Provisional Sprint Contract

Recommended next execution slice: **Slice 1: metric and dataset schema
hardening**, preceded by the Slice 0 contract tests.

Goal:

- Remove the highest-risk contract drift before moving paths or adding a public
  API.

In-scope work:

- metric selector adapter
- dataset-kind validation split
- plot precondition split
- `combine_aa` canonical metric consumption
- docs alignment for metric columns and `RECORD.md`
- temp-output smoke test for one small DMS path

Out-of-scope work:

- full workspace migration
- protein-native DMS API
- large job/output directory reorganization
- generated result commits

Done criteria:

- existing 56 Permuter tests pass
- new drift-reproduction tests pass
- strict validation has a documented rule for `epistasis` or its namespaced
  replacement
- current jobs still validate
- docs checks pass

Verification:

```bash
uv run pytest -q src/dnadesign/permuter/tests
uv run permuter --help
uv run python -m dnadesign.devtools.docs.checks --repo-root .
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run ruff check src/dnadesign/permuter
uv run ruff format --check src/dnadesign/permuter
git diff --check
```

## Risk Handling

- Avoid broad rewrites. Use adapter seams first, then migrate callers.
- Keep legacy read compatibility for existing datasets, but do not let legacy
  columns remain the write contract.
- Do not add cross-tool shared schemas until another tool consumes the contract.
- Do not make workspace support depend on generated package-local `results/`.
- Fail fast on invalid layouts, missing dependencies, duplicate run ids, and
  ambiguous metric ids.
- Keep read-only commands read-only by default once pure mode exists.

## Open Questions

1. Should new interaction metrics use only the namespaced form
   `permuter__interaction__epistasis__<metric_id>`, or should `epistasis`
   remain an accepted canonical unprefixed exception?
2. For future protein DMS, should the first public API be protein-native only,
   or should the first version require coding DNA so Evo2-DNA scoring is
   immediately supported?
3. Should checked-in workspace examples live under
   `src/dnadesign/permuter/workspaces/`, or should Permuter begin with
   template workspaces only and keep live study workspaces in study-owned docs?
