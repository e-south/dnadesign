---
doc_id: architecture
surface: system-of-record
owner: dnadesign-maintainers
last_verified: 2026-10-29
---

# ARCHITECTURE

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-29

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
- Top-level `src/dnadesign/` contains supported tool packages and shared infrastructure (`artifacts`, `contracts`, `devtools`). Retired source is excluded from the runtime namespace.
- Runbooks and references: `docs/`
- CI/test/devtool orchestration: `.github/workflows/ci.yaml` and `src/dnadesign/devtools/`
- Package/dependency contracts: `pyproject.toml`, `uv.lock`, `pixi.toml`, `pixi.lock`

## System boundaries
- Tool packages: each top-level tool under `src/dnadesign/<tool>/` owns its CLI behavior, configs, and tests.
- Domain-qualified tools may own reusable biological operations while callers
  retain study identity, candidate choices, and experimental evidence.
- Shared artifact schemas live under `src/dnadesign/contracts/` when a producer and consumer need a neutral, versioned handoff model without importing either tool's internals.
- Construct accepts producer-authored annotated sequence parts through the
  neutral `AnnotatedSequencePartV1` contract. It may place and offset that
  object, but it does not rederive producer-owned nested features.
- Shared create-only filesystem publication mechanics live under `src/dnadesign/artifacts/`; producing tools still own their artifact schemas and manifest meaning.
- Shared test infrastructure lives under `src/dnadesign/devtools/tests/support/` and is test-only by contract; production code must not depend on it.
- OPS core is a neutral shell around discovery, observation/status, orchestration,
  and generic readiness evaluation; it must not own sibling-specific provider
  implementations or study-specific status/preflight policy.
- Shared operational plane: Notify (`src/dnadesign/notify/`) consumes SeqLedger events as integration signals without controlling producer tools.
- Shared storage semantics: SeqLedger overlay/compaction/file-shape contracts use SeqLedger domain terms and stay tool-agnostic so Construct, Infer, and external producers can share one records store.
- Shared developer infrastructure: devtools modules provide repository-only CI scope detection, docs checks, coverage gates, and quality entropy reporting. Installed command launchers live in `src/dnadesign/_entrypoints.py` and do not import devtools.
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
- `src/dnadesign/seqledger/docs/operations/` is the default home for durable cross-tool data-plane procedures when the shared handoff artifact is a SeqLedger dataset, overlay set, or `.events.log`.

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
- Live studies are external clients. They own their scientific intent,
  configurations, evidence, and decisions, and call versioned dnadesign APIs
  through explicit workspace paths. Dnadesign does not select a global active
  study or infer one from its repository layout.
- Tool packages own their workload configs, runtime outputs, and package-local workspace templates.
- Private or large runtime objects live outside public Git checkouts under the
  neutral `dnadesign.storage-object/v1` envelope. The envelope distinguishes
  workspaces, durable stores, and rebuildable tool caches while closing
  identity, ownership, retention, paths, and digests. The producing tool
  continues to own each object's internal schema and behavior.
- SeqLedger owns durable dataset records and the integration event stream (`.events.log`) that downstream tooling consumes.
- Active shared SeqLedger dataset ids are flat owner-first contracts, for example
  `construct_library`; provenance belongs in `root_kind`,
  `owner_tool`, overlays, event metadata, and study records rather than nested
  tool-routing folders.
- Cross-dataset SeqLedger overlay transfer is explicit-only: maintenance merge defaults to base-row merge, while any overlay carry must be opt-in, namespace-scoped, schema-compatible, and auditable in events.
- Workspaces that enable SeqLedger sinks must select an external dataset root explicitly.
  SeqLedger's standalone default is `./usr_datasets`; installed package resources are
  not writable dataset locations.
- Workspace-local export roots remain allowed only as explicit opt-in producer
  mirrors, self-contained demos, or handoff surfaces named directly in a study
  record or runbook.
- Cross-tool coupling is file/event contract based; packages must not depend on internal `src.*` modules across tool boundaries.
- Import checks cover runtime packages and shared contracts, including `src/workspaces` and `src/notebooks`. Tool-root data and example shelves are resources, not runtime source.
- Utility modules must stay tool-local (`src/dnadesign/<tool>/...`); top-level shared `src/dnadesign/utils` is not an allowed boundary.
- Study status and preflight logic is study-owned once it becomes specific.
  External packages register status metadata through the
  `dnadesign.ops.status_registries` entry-point group. OPS imports only the
  selected provider; dnadesign does not contain study-specific execution
  taxonomies.
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
- Producer tools -> SeqLedger event stream (`.events.log`) -> observer tools and webhook sinks.
- Tool outputs -> dataset artifacts or workspace outputs -> downstream analysis or optimization tools.
- Developer workflow -> core CI lane (lint/docs/standard tests + coverage gate) and external integration lane (FIMO/integration) when required.

## Architecture invariants
- Component support status is declared in `.github/tool-lifecycle.toml`.
  Active and maintenance components are installed unless declared
  `installed = false` for repository-only tooling. Retired components must not
  enter source, commands or distributions.
  Catalog checks use these declarations. Directory discovery identifies
  unclassified source rather than granting it supported status.
  `pyproject.toml` owns entrypoints, dependencies and runtime resources.
  Installed tool commands do not read repository lifecycle metadata.
- No silent fallbacks: missing required inputs/dependencies must fail fast with actionable errors.
- Boundary contracts are explicit: CI scope, tool coverage baselines, and marker-based external integration tests are enforced in code.
- Docs are layered: root docs are maps and deep procedures stay in runbooks/reference docs.
- Cross-tool deep procedures must have one authoritative location with root-doc routing; do not duplicate the same operator sequence in multiple tool trees.
- Cross-tool placement is ownership-driven:
  - use `docs/operations/` when the owner is scheduler/audit/log sequencing
  - use `src/dnadesign/seqledger/docs/operations/` when the owner is a durable SeqLedger dataset, overlay set, or `.events.log`
  - use the downstream tool docs after the handoff when that tool owns the next state machine
- Cross-tool path ownership is explicit: repeated runs accumulate in workspace-scoped directories, not repository-root runbook/log fan-out.
- Repository-root generated artifact and transient operational directories (for example `outputs/`, `.codex_tmp/`, `.tmp_ops/`, `tmp_ops/`) are disallowed by policy; disposable working state belongs under `/scratch` and durable state belongs under a tool or study workspace root.

## Where to go deeper
- Maintainer index: `docs/dev/README.md`
- BU SCC operator references: `docs/bu-scc/README.md`, `docs/bu-scc/setup/quickstart.md`, `docs/bu-scc/runbooks/batch-notify.md`, and `docs/bu-scc/jobs/README.md`
- Notify event contract: `docs/notify/seqledger-events.md`
- Reliability operations: `RELIABILITY.md`
- Security policy and secrets handling: `SECURITY.md`
- Engineering invariants: `DESIGN.md`
- Active/proposed work tracking: `PLANS.md`
