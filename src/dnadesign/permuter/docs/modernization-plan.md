# Permuter Contract Modernization

**Type:** implementation status and remaining-work contract
**Owner:** dnadesign-maintainers
**Owner-boundary:** `src/dnadesign/permuter`
**Status:** core contract slices implemented
**Last verified:** 2026-07-14

## Outcome

Permuter is the public variant-generation and in silico DMS surface for DNA,
protein, and coding-DNA-backed requests. It has a stable public API,
workspace-scoped CLI execution, explicit dataset identity, namespaced metric
columns, and a non-executing Infer feature-request handoff.

The package owns variant intent and materialization mechanics. It does not own
USR synchronization, Construct placement, study interpretation, Infer feature
sidecars, or Ops run-state semantics.

## Current Architecture

```text
src/dnadesign/permuter/
  __init__.py                    public facade and CLI entrypoint
  src/
    api/                         typed generation, evaluation, materialization,
                                 validation, and Infer request contracts
    contracts/                   metric-column contracts
    cli/                         Typer parsing and output adapters
    evaluators/                  pluggable scoring implementations
    plots/                       registered plot implementations
    protocols/                   mutation protocols
    workspaces/                  workspace loading and validation
    resources/                   packaged codon tables
  workspaces/
    _shared/inputs/              shared checked-in inputs
    <scope>/config.yaml          one runnable scope
    <scope>/outputs/             generated artifacts
```

Cross-tool callers import `dnadesign.permuter`. They do not import
`dnadesign.permuter.src.*` or CLI handlers.

## Implemented Contracts

### Public generation and materialization

The public facade exports:

- `NucleotideDmsRequest`
- `ProteinDmsRequest`
- `CodingDnaDmsRequest`
- `EvaluatorPlan`
- `MetricSpec`
- `VariantRecord`
- `PermuterResult`
- `DatasetRef`
- `ValidationReport`
- `generate_variants(...)`
- `evaluate_variants(...)`
- `materialize_result(...)`
- `validate_dataset(...)`

Pure generation returns typed in-memory records. Filesystem materialization is
an explicit second operation. Coding-DNA requests require an explicit codon
table and may set `max_variants` to reject oversized scans before rows are
materialized.

Materialized identity is deliberately split:

- `id` is the canonical USR sequence identity derived from `bio_type` and
  `sequence`.
- `permuter__var_id` is the Permuter request/variant identity preserved from
  `VariantRecord.id`.

`metadata.permuter.variant_id` and a parallel `permuter__variant_id` column are
rejected. A new identity spelling requires a versioned materialization
contract, not a second column.

### Metric columns

Canonical metric columns are:

- observed: `permuter__observed__<metric_id>`
- expected: `permuter__expected__<metric_id>`
- interaction: `permuter__interaction__<interaction_id>__<metric_id>`

Fixed-vector evaluator values may use Arrow list columns or declared observed
subcolumns. `permuter__metric__*` is unsupported and rejected by strict
validation; the runtime does not carry a read alias.

### Workspace execution

Each scope has exactly one `config.yaml`. Scope name, input reference columns,
protocol, and output layout are validated before execution. Output layouts are
`flat` or `nested`; every other value fails validation.

The resolver accepts a workspace directory, a `config.yaml` path, or a scope
ID. It supports `${WORKSPACE_DIR}`, `${WORKSPACES_DIR}`,
`${PERMUTER_RESOURCE_DIR}`, environment variables, and `~`. `${JOB_DIR}` is
unsupported. `--out` is the explicit output override; otherwise
`PERMUTER_OUTPUT_ROOT` may provide a federated writable root. The runtime does
not probe undeclared output layouts.

The checked-in workspace configurations and shared input CSVs are included as
`dnadesign.permuter` package data.

### CLI and read-only behavior

The CLI provides:

- `permuter run`
- `permuter evaluate`
- `permuter plot`
- `permuter export`
- `permuter validate`
- `permuter inspect`
- `permuter workspace list|validate|inspect`

`run`, `evaluate`, `plot`, and `validate` expose one-object JSON output for
automation. `validate` and `inspect` do not append to `RECORD.md` unless the
caller passes `--record`.

Plot runs write `plots/manifest.json` with schema
`permuter.plot_manifest.v1`, source provenance, metric and render parameters,
and emitted artifacts. Only plot IDs in the registry are advertised by the
CLI.

### Infer boundary

Permuter exposes two distinct Infer-facing surfaces:

1. Evo2 evaluators call the public `dnadesign.infer.run_extract` facade to
   compute candidate-level scores.
2. `InferFeatureRequest` records a non-executing feature-bundle request with an
   explicit source dataset, feature-bundle reference, sequence-view selectors,
   and requested outputs.

Neither surface makes Permuter the owner of Infer feature aliases,
vector/scalar sidecars, completion ledgers, resume planning, stale detection,
or `_derived/infer` writes. Candidate scores must not be presented as evidence
that those Infer artifacts exist.

## Remaining Explicit Gap

There is no Permuter executor for `InferFeatureRequest`. Add one only when a
real consumer requires a Permuter-initiated, sidecar-native feature run. That
slice must:

- call public Infer APIs only;
- preserve Infer feature-bundle and sequence-view contracts;
- keep execution and writeback ownership with Infer;
- prove feature alias and scalar/vector sidecar behavior with a smoke workspace;
- keep study overlay policy outside Permuter;
- avoid generated workspace artifacts in tests.

Until then, use the request manifest as a handoff and run the owning Infer
workflow directly.

## Maintainer Decisions

- Interaction metrics use the namespaced interaction form; unprefixed
  exceptions are not part of the contract.
- Protein-native and coding-DNA-backed DMS are separate explicit request types.
- Checked-in runnable examples remain workspace scopes under
  `src/dnadesign/permuter/workspaces/`.
- Generated datasets remain workspace-Parquet artifacts. USR event logs,
  overlay synchronization, and Infer sidecar reuse belong to their owning
  tools.
- Unsupported protocol, metric, identity, and layout names fail; do not add
  silent aliases or fallback probes.

## Verification

```bash
uv run pytest -q src/dnadesign/permuter/tests
uv run permuter --help
uv run permuter workspace list --root src/dnadesign/permuter/workspaces
uv run python -m dnadesign.devtools.docs.checks --repo-root .
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run ruff check src/dnadesign/permuter
uv run ruff format --check src/dnadesign/permuter
git diff --check
```
