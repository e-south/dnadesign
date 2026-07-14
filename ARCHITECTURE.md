---
doc_id: architecture
surface: system-of-record
owner: dnadesign-maintainers
last_verified: 2026-07-13
---

# ARCHITECTURE

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13

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
- Top-level `src/dnadesign/` is a controlled namespace: boundary-owning packages plus shared infrastructure (`contracts`, `devtools`) and reserved legacy buckets (`archived`, `prototypes`) only.
- Runbooks and references: `docs/`
- CI/test/devtool orchestration: `.github/workflows/ci.yaml` and `src/dnadesign/devtools/`
- Package/dependency contracts: `pyproject.toml`, `uv.lock`, `pixi.toml`, `pixi.lock`

## System boundaries
- Tool packages: each top-level tool under `src/dnadesign/<tool>/` owns its CLI behavior, configs, and tests.
- Shared artifact schemas live under `src/dnadesign/contracts/` when a producer and consumer need a neutral, versioned handoff model without importing either tool's internals.
- Shared test infrastructure lives under `src/dnadesign/devtools/tests/support/` and is test-only by contract; production code must not depend on it.
- OPS core is a neutral shell around discovery, observation/status, orchestration,
  and generic readiness evaluation; it must not own sibling-specific provider
  implementations or study-specific status/preflight policy.
- Shared operational plane: Notify (`src/dnadesign/notify/`) consumes USR events as integration signals without controlling producer tools.
- Shared storage semantics: USR overlay/compaction/file-shape contracts use USR domain terms and stay tool-agnostic so DenseGen, Infer, and future producers can share one records store.
- Shared developer infrastructure: devtools modules provide CI scope detection, docs checks, coverage gates, and quality entropy reporting.
- Operator-facing HPC boundary: BU SCC guidance and templates live in `docs/bu-scc/`.

## Document authority order
- `ARCHITECTURE.md` is the top-level authority for cross-tool boundaries and path ownership contracts.
- `DESIGN.md` defines implementation invariants that must remain consistent with architecture boundaries.
- `RELIABILITY.md` and `SECURITY.md` specialize runtime and secret-handling policy without overriding architecture boundaries.
- `docs/operations/orchestration/runbooks.md` is the executable operator contract for batch orchestration behavior.
- `PLANS.md` governs lifecycle/process for promoting or changing these contracts.
- Root docs route readers to one authoritative deep procedure; cross-tool runbooks may live either in top-level `docs/` or in the boundary-owning tool's operations docs when that tool owns the durable handoff.
- `docs/runbooks/README.md` is the centralized inventory of authoritative cross-tool procedures and tool-local runbook sources; it is a discovery surface, not the owner of those procedures.
- `docs/operations/` is the root control-plane orchestration surface only; it is not the generic registry for durable cross-tool data-plane workflows.
- `src/dnadesign/usr/docs/operations/` is the default home for durable cross-tool data-plane procedures when the shared handoff artifact is a USR dataset, overlay set, or `.events.log`.

## Cross-tool information architecture
- Plane semantics are explicit and non-interchangeable:
  - discovery plane: `ops catalog`
  - observation plane: `ops progress`
  - control plane: `ops runbook`
  - record plane: checked-in study records and other checked-in state
  - execution-readiness plane: deeper preflight blockers on the current host,
    workspace, or cluster
  - data plane: dataset and artifact posture
- Workspace-rooted accumulation is the contract for repeated campaigns; orchestration state must not fan out into repository-root ad-hoc files.
- Root `docs/README.md` is the only top-level router. It routes by user intent and ownership plane, then hands off to exactly one authoritative deep procedure for each cross-tool workflow.
- `docs/runbooks/README.md` is the concise inventory surface for authoritative procedures; it links to owner-local runbooks and workflows without relocating them.
- Control-plane orchestration artifacts stay under workspace-scoped logging roots; tool-local docs define exact subpaths and artifact names.
- Boundary-owned observation surfaces publish checked-in metadata under
  tool-local `src/dnadesign/**/ops/status.registry.yaml` files. Ops-owned
  built-in providers live under `src/dnadesign/ops/providers/*/status.registry.yaml`.
  OPS recursively discovers those fragments, renders help from metadata alone,
  and imports provider code only for the selected surface.
- Checked-in study records are study-first rather than family-nested:
  `docs/studies/index.yaml` selects the active study, each live study record
  lives under `docs/studies/<study-id>/`, and Ops-facing routes are declared
  explicitly with `ops_surfaces.status_kind` and `ops_surfaces.preflight_kind`
  in `ops.study.yaml`.
- Tool packages own their workload configs, runtime outputs, and package-local workspace templates.
- USR owns durable dataset records and the integration event stream (`.events.log`) that downstream tooling consumes.
- Active shared USR dataset ids are flat owner-first contracts, for example
  `densegen_prom_eth_cip_source`; provenance belongs in `root_kind`,
  `owner_tool`, overlays, event metadata, and study records rather than nested
  tool-routing folders.
- Cross-dataset USR overlay transfer is explicit-only: maintenance merge defaults to base-row merge, while any overlay carry must be opt-in, namespace-scoped, schema-compatible, and auditable in events.
- Curated study-facing workspaces that enable USR sinks should default those
  sinks to an explicit shared USR root such as `src/dnadesign/usr/datasets`.
- Workspace-local export roots remain allowed only as explicit opt-in producer
  mirrors, self-contained demos, or handoff surfaces named directly in a study
  record or runbook.
- Cross-tool coupling is file/event contract based; packages must not depend on internal `src.*` modules across tool boundaries.
- Utility modules must stay tool-local (`src/dnadesign/<tool>/...`); top-level shared `src/dnadesign/utils` is not an allowed boundary.
- Study status and preflight logic is study-owned once it becomes specific.
  OPS discovers provider metadata and imports only the selected provider
  entrypoint; study-specific execution taxonomy stays under
  `src/dnadesign/studies/units/<study-id>/`.
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
- If extra route nuance is needed, use a separate field such as `Surface role`
  or `Relationship`; do not invent replacement plane names such as
  `producer-analysis`, `execution-surface`, or `downstream-analysis`.

## High-level data flows
- Producer tools -> USR event stream (`.events.log`) -> observer tools and webhook sinks.
- Tool outputs -> dataset artifacts or workspace outputs -> downstream analysis or optimization tools.
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
- Repository-root generated artifact and transient operational directories (for example `outputs/`, `.codex_tmp/`, `.tmp_ops/`, `tmp_ops/`) are disallowed by policy; disposable working state belongs under `/scratch` and durable state belongs under a tool or study workspace root.

## Where to go deeper
- Maintainer index: `docs/dev/README.md`
- Monorepo organization audit: `docs/dev/audits/monorepo-organization.md`
- BU SCC operator references: `docs/bu-scc/README.md`, `docs/bu-scc/setup/quickstart.md`, `docs/bu-scc/runbooks/batch-notify.md`, and `docs/bu-scc/jobs/README.md`
- Notify event contract: `docs/notify/usr-events.md`
- Reliability operations: `RELIABILITY.md`
- Security policy and secrets handling: `SECURITY.md`
- Engineering invariants: `DESIGN.md`
- Active/proposed work tracking: `PLANS.md`
