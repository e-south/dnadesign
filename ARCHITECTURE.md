# ARCHITECTURE

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-03

## At a glance
`dnadesign` is a uv-managed monorepo of modular bioinformatics tools under `src/dnadesign/`, with shared CI/devtools and operator runbooks in `docs/`.
This file is the architecture map: it names system boundaries, major flows, and invariants, then links to deeper operational/reference docs.

## Contents
- [Repository shape](#repository-shape)
- [System boundaries](#system-boundaries)
- [Document authority order](#document-authority-order)
- [Cross-tool information architecture](#cross-tool-information-architecture)
- [High-level data flows](#high-level-data-flows)
- [Architecture invariants](#architecture-invariants)
- [Where to go deeper](#where-to-go-deeper)

## Repository shape
- Code: `src/dnadesign/`
- Runbooks and references: `docs/`
- CI/test/devtool orchestration: `.github/workflows/ci.yaml` and `src/dnadesign/devtools/`
- Package/dependency contracts: `pyproject.toml`, `uv.lock`, `pixi.toml`, `pixi.lock`

## System boundaries
- Tool packages: each top-level tool under `src/dnadesign/<tool>/` owns its CLI behavior, configs, and tests.
- Shared operational plane: Notify (`src/dnadesign/notify/`) consumes USR events as integration signals without controlling producer tools.
- Shared storage semantics: USR overlay/compaction/file-shape contracts use USR domain terms and stay tool-agnostic so DenseGen, Infer, and future producers can share one records store.
- Shared developer infrastructure: devtools modules provide CI scope detection, docs checks, coverage gates, and quality entropy reporting.
- Operator-facing HPC boundary: BU SCC guidance and templates live in `docs/bu-scc/`.

## Document authority order
- `ARCHITECTURE.md` is the top-level authority for cross-tool boundaries and path ownership contracts.
- `DESIGN.md` defines implementation invariants that must remain consistent with architecture boundaries.
- `RELIABILITY.md` and `SECURITY.md` specialize runtime and secret-handling policy without overriding architecture boundaries.
- `docs/operations/orchestration-runbooks.md` is the executable operator contract for batch orchestration behavior.
- `PLANS.md` governs lifecycle/process for promoting or changing these contracts.
- Root docs route readers to one authoritative deep procedure; cross-tool runbooks may live either in top-level `docs/` or in the boundary-owning tool's operations docs when that tool owns the durable handoff.
- `docs/operations/` is the root control-plane orchestration surface only; it is not the generic registry for durable cross-tool data-plane workflows.
- `src/dnadesign/usr/docs/operations/` is the default home for durable cross-tool data-plane procedures when the shared handoff artifact is a USR dataset, overlay set, or `.events.log`.

## Cross-tool information architecture
- Workspace-rooted accumulation is the contract for repeated campaigns; orchestration state must not fan out into repository-root ad-hoc files.
- Root `docs/README.md` is the only top-level router. It routes by user intent and ownership plane, then hands off to exactly one authoritative deep procedure for each cross-tool workflow.
- `ops` owns orchestration artifacts under `<workspace-root>/outputs/logs/ops/`:
  - runbooks: `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml`
  - audit trail: `<workspace-root>/outputs/logs/ops/audit/latest.json`
  - scheduler stdout: `<workspace-root>/outputs/logs/ops/sge/<runbook-id>/`
  - runtime traces: `<workspace-root>/outputs/logs/ops/runtime/`
- `densegen` and `infer` own workload execution using `<workspace-root>/config.yaml` and write domain outputs under workspace outputs/tables and dataset materialization paths.
- `usr` owns dataset records and the integration event stream (`.events.log`) that downstream tooling consumes.
- Cross-dataset USR overlay transfer is explicit-only: maintenance merge defaults to base-row merge, while any overlay carry must be opt-in, namespace-scoped, schema-compatible, and auditable in events.
- Curated dnadesign workspaces default USR dataset roots to `<workspace-root>/outputs/usr_datasets`.
- Explicit external USR roots remain allowed for sync and mirror workflows when the operator chooses them deliberately.
- `notify` owns delivery wiring under `<workspace-root>/outputs/notify/<tool>/` and consumes USR events without mutating DenseGen/Infer domain artifacts.
- Cross-tool coupling is file/event contract based; packages must not depend on internal `src.*` modules across tool boundaries.
- Utility modules must stay tool-local (`src/dnadesign/<tool>/...`); top-level shared `src/dnadesign/utils` is not an allowed boundary.
- Document-type semantics are explicit:
  - `route`: index entry or decision surface only
  - `runbook`: authoritative operator procedure with ordered commands and verification
  - `workflow`: downstream tool-owned branch or state-machine procedure when a tool intentionally uses a `workflows/` subtree
  - `tutorial`: pedagogical walkthrough, not the authority surface
  - `demo`: packaged sample assets or tracer-bullet workspace/profile, not the authority surface
- Authoritative runbook and workflow semantics must not rely on back-compat shims:
  - deprecated ids, commands, and document-type labels are removed from the supported surface
  - if a rename is necessary, fail fast with an actionable error instead of silently normalizing
- Cross-tool deep procedures must declare:
  - `Type`
  - `Plane`
  - `Owner-boundary`
  - `Entry artifact`
  - `Exit artifact`
- `Plane` values are limited to the ownership planes in this repo:
  - `control-plane`
  - `data-plane`
  - `downstream-tool`

## High-level data flows
- DenseGen/other producers -> USR event stream (`.events.log`) -> Notify watcher/webhook sink.
- Tool outputs -> dataset artifacts (for example Parquet tables) -> downstream analysis tools (cluster, billboard, nmf, latdna, cruncher, tfkdanalysis).
- Developer workflow -> core CI lane (lint/docs/standard tests + coverage gate) and external integration lane (FIMO/integration) when required.

## Architecture invariants
- No silent fallbacks: missing required inputs/dependencies must fail fast with actionable errors.
- Boundary contracts are explicit: CI scope, tool coverage baselines, and marker-based external integration tests are enforced in code.
- Docs are layered: root docs are maps and deep procedures stay in runbooks/reference docs.
- Cross-tool deep procedures must have one authoritative location with root-doc routing; do not duplicate the same operator sequence in multiple tool trees.
- Cross-tool placement is ownership-driven:
  - use `docs/operations/` when the owner is scheduler/audit/log sequencing
  - use `src/dnadesign/usr/docs/operations/` when the owner is a durable USR dataset, overlay set, or `.events.log`
  - use the downstream tool docs after the handoff when that tool owns the next state machine
- Cross-tool path ownership is explicit: repeated runs accumulate in workspace-scoped directories, not repository-root runbook/log fan-out.
- Repository-root transient operational working directories (for example `.codex_tmp/`, `.tmp_ops/`, `tmp_ops/`) are disallowed by policy; disposable working state belongs under `/scratch` and durable state belongs under `<workspace-root>/outputs/logs/ops/`.

## Where to go deeper
- Maintainer index: `docs/dev/README.md`
- Monorepo organization audit: `docs/dev/monorepo-organization-audit.md`
- BU SCC operator references: `docs/bu-scc/README.md`, `docs/bu-scc/quickstart.md`, `docs/bu-scc/batch-notify.md`, and `docs/bu-scc/jobs/README.md`
- Notify event contract: `docs/notify/usr-events.md`
- Reliability operations: `RELIABILITY.md`
- Security policy and secrets handling: `SECURITY.md`
- Engineering invariants: `DESIGN.md`
- Active/proposed work tracking: `PLANS.md`
