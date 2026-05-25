# Permuter Modernization Plan

**Type:** living dev spec
**Owner:** dnadesign-maintainers
**Owner-boundary:** `src/dnadesign/permuter`
**Status:** active
**Last verified:** 2026-05-24

## Plan Intent Summary

Modernize Permuter from a CLI/scope-first mini-project into a contract-backed tool
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
older metric-column drift, newer observed/expected columns, RT-specific selection
logic, and package-local generated outputs. Hardening those seams now reduces
future maintenance cost and makes Permuter safe to call from studies, notebooks,
cluster runs, and other `dnadesign` tools.

## Current Audit Baseline

Verified on 2026-05-24:

- `uv run permuter --help` passed.
- `uv run pytest -q src/dnadesign/permuter/tests` passed with 94 tests after
  the public API/JSON/plot-manifest/SCC wrapper slice.
- All 9 checked-in workspace scope configs validated with `ScopeConfig`.
- Scratch dogfood `run -> evaluate -> validate -> plot` passed and rendered
  PNG/PDF plot artifacts.

Primary findings:

- High: Permuter no longer reaches into stale Infer internals, but its Evo2
  evaluators still use the ad hoc evaluator path. They do not yet opt into
  Infer's current feature-bundle, sequence-view alias, scalar/vector sidecar,
  `_derived/infer`, or USR overlay contracts.
- Medium: generated datasets remain workspace-Parquet-native. They are
  USR-shaped rows, but not overlay-native sidecars with `.events.log`,
  sync/diff semantics, or Infer sidecar reuse.
- Medium: public generation is clean, but scoring, materialization, and
  validation needed stable library APIs so sibling tools do not shell out or
  import `dnadesign.permuter.src.*`.
- Medium: `run`, `evaluate`, `plot`, and `validate` needed one-object JSON
  output so naive agents and SCC wrappers can coordinate without scraping rich
  console text.
- Low/medium: plot IDs drifted. `window_score_mass` appeared in help/config
  sizing surfaces even though CLI dispatch rejected it. Plot artifacts also
  lacked a freshness manifest.
- Low: BU SCC had Infer/Evo2 templates but no Permuter wrapper for
  closed-loop evaluate jobs.

## Explicit Scope

In scope:

- dataset column contract modernization
- scope config parsing and schema validation hardening
- workspace model introduction
- public API facade for generation and evaluation planning
- protein-native or codon-policy-backed DMS request design
- docs and tests that make the contracts executable
- migration of former runnable configs into workspace scopes

Out of scope for the first modernization pass:

- replacing Evo2 evaluator internals
- changing USR core semantics
- moving cross-tool schemas into shared contracts before a real cross-tool
  consumer needs them
- rewriting all protocols at once
- preserving deprecated identifiers as silent aliases
- committing generated workspace `outputs/` artifacts

## Target Architecture

### Boundary Model

Permuter should have four explicit layers:

1. `api`
   Public, typed library entrypoints. No console output, no implicit filesystem
   writes unless the caller passes a workspace or output policy.

2. `contracts`
   Pydantic/dataclass request and result objects, dataset schema helpers,
   and metric-column selectors.

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
  config.yaml
  outputs/
    records.parquet
    plots/
```

Workspace principles:

- `config.yaml` is the entry artifact for workspace-backed execution.
- `outputs/` is generated and ignored unless explicitly promoted.
- Downstream dependencies between DMS, combine, and selection runs are explicit
  workspace-relative paths, not hard-coded paths into package-root `results/`.
- The CLI uses `--workspace`; package-root `jobs/`, `inputs/`, `results/`, and
  `notebooks/` are not part of the architecture.

### Dataset Column Contract

Canonical columns:

- USR core: `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`,
  `created_at`
- variant identity: `permuter__var_id`
- generation metadata: `permuter__scope`, `permuter__ref`,
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
- `MetricSpec`
- `DatasetRef`
- `VariantRecord`
- `PermuterResult`
- `ValidationReport`

Initial API functions:

- `generate_variants(request) -> PermuterResult`
- `evaluate_variants(result, plan) -> PermuterResult`
- `materialize_result(result, output_dir) -> DatasetRef`
- `validate_dataset(dataset, strict=False) -> ValidationReport`

Protein DMS policy:

- Protein-native enumeration should emit protein variant records without
  pretending a nucleotide sequence exists.
- If DNA or Evo2-DNA scoring is requested from a protein request, the caller must
  provide a coding DNA reference or an explicit codon design policy.
- Back-translation must be explicit and recorded in provenance.

Current implementation note:

- `generate_variants(...)` is implemented for `NucleotideDmsRequest`,
  `ProteinDmsRequest`, and `CodingDnaDmsRequest`.
- `CodingDnaDmsRequest` is the construct-ready DMS surface: it accepts a coding
  DNA reference plus an explicit codon table and returns DNA variant records
  with codon-policy provenance in metadata.
- `evaluate_variants(...)`, `materialize_result(...)`, and
  `validate_dataset(...)` are public surfaces. Sibling tools should call the
  facade at `dnadesign.permuter` and should not import CLI functions or
  `dnadesign.permuter.src.*` modules as stand-ins.
- The public scoring path shares evaluator output normalization with the CLI:
  scalar, dict, and fixed-vector evaluator outputs become canonical
  `permuter__observed__*` columns when materialized.
- Public API materialization now keeps USR sequence identity and Permuter
  variant identity separate: materialized `id` is the canonical USR sequence id,
  while `VariantRecord.id` is preserved as `permuter__var_id`.
- This API is still not a substitute for the future Infer feature-bundle/USR
  sidecar bridge. Evo2 feature bundles should use Infer directly until Permuter
  has an explicit sidecar-native evaluator mode.

### Machine and Batch Surface

- `permuter run --json`, `permuter evaluate --json`,
  `permuter validate --json`, and `permuter plot --json` emit a single JSON
  object on stdout.
- Plot runs write `plots/manifest.json` with source parquet path, source size,
  source mtime, metric id, render parameters, and emitted artifacts.
- Supported plot IDs are centralized in `plots.registry`. Internal helpers such
  as `window_score_mass` are not advertised unless they have a complete CLI
  renderer contract.
- BU SCC has a `docs/bu-scc/jobs/permuter-evaluate.qsub` wrapper that validates
  a workspace, optionally runs generation, evaluates with JSON output, and
  writes a workspace-scoped runtime trace.

### Next Architecture Gap

Permuter's next high-value slice is an explicit Infer bridge:

- Add a non-executing handoff manifest that references an Infer workspace or
  feature-bundle plan instead of an ad hoc Evo2 output spec.
- Preserve Infer-owned sidecar contracts rather than copying private
  implementation details into Permuter.
- Decide whether Permuter writes a USR overlay namespace directly or returns a
  handoff object that an owning study/workspace syncs into USR.
- Keep the fast ad hoc Evo2 evaluator path explicit, named, and documented so it
  is not mistaken for full Infer sidecar semantics.

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
- Treat each former runnable config as its own workspace scope with `config.yaml`.
- Implement `PERMUTER_OUTPUT_ROOT` or remove it from docs and help. Hidden
  fallback behavior is not allowed.
- Remove retired dataset-layout probes. `evaluate` and `plot` must resolve the
  configured workspace/ref path and fail if that path is missing.
- Make invalid `output.layout` values fail fast instead of falling through to
  nested layout.

Done when:

- workspace validation fails fast on missing inputs, scope/config name mismatch,
  and invalid layouts
- workspace output paths stay under workspace `outputs/` unless an explicit
  external root is configured
- workspace CLI resolves scope ids, workspace directories, and `config.yaml`
  paths

### Slice 4: Public API Facade

- Add `dnadesign.permuter` public package facade with typed request/result objects.
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

Recommended next execution slice: **Infer feature-bundle and USR-sidecar
bridge**.

Goal:

- Let Permuter evaluate generated variants through Infer's current public
  feature-bundle/sidecar semantics without internal Infer inreach.

In-scope work:

- public `InferScoringPlan` or equivalent handoff contract owned by Permuter's
  facade
- an implementation path that calls Infer public APIs only
- feature alias and sequence-view contract tests against an Infer smoke
  workspace
- scalar/vector sidecar preservation tests
- explicit documentation that distinguishes ad hoc Evo2 scoring from
  sidecar-native Infer scoring

Out-of-scope work:

- copying Infer private config/runtime internals into Permuter
- adding compatibility shims for old Infer workspaces
- making Permuter own study-specific USR overlay policy
- committing generated workspace artifacts

Done criteria:

- Permuter tests pass
- Infer feature-bundle smoke validation and dry-run pass
- architecture boundary checks pass
- a temp-output Permuter dogfood path can choose either ad hoc scoring or
  sidecar-native Infer scoring by explicit plan
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
- Do not let legacy columns remain the write contract.
- Do not add cross-tool shared schemas until another tool consumes the contract.
- Do not make workspace support depend on generated package-root `results/`.
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
