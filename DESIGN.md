---
doc_id: design
surface: system-of-record
owner: dnadesign-maintainers
last_verified: 2026-07-14
---

# DESIGN

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

## At a glance
This document defines repo-wide engineering principles, invariants, and boundary rules.
Use it when adding or changing behavior so tools remain decoupled, assertive, and easy to evolve.

## Contents
- [Core principles](#core-principles)
- [Boundary contracts](#boundary-contracts)
- [Information architecture invariants](#information-architecture-invariants)
- [Tool/package boundaries](#toolpackage-boundaries)
- [Documentation model](#documentation-model)
- [References](#references)

## Core principles
- Parse, don't validate: parse at boundaries, then operate on trusted structures internally.
- Fail fast on invalid or missing prerequisites; avoid silent fallback behavior.
- Keep units cohesive and interfaces explicit.
- Prefer simple, testable composition over implicit cross-module coupling.

## Boundary contracts
- CLI/config inputs must be parsed into explicit structures before execution.
- Parsing belongs at tool boundaries (for example tool CLI/config loading modules), so downstream logic can assume invariant-preserving inputs.
- CI contracts are code-backed, not convention-backed:
  - marker semantics (`fimo`, `integration`)
  - changed-file scope detection
  - per-tool coverage baseline gates
- External-tool dependencies (for example FIMO/MEME) are checked explicitly before heavy execution paths.

## Information architecture invariants
- For repeated batch attempts, orchestration state is workspace-scoped by default:
  - runbook file: `<workspace-root>/outputs/logs/ops/runbooks/<runbook-id>.yaml`
  - audit file: `<workspace-root>/outputs/logs/ops/audit/latest.json`
- Tool run-mode behavior must stay explicit and fail-fast; resume/fresh transitions may not be inferred from ambiguous state.
- Cross-tool output ownership is orthogonal:
  - control-plane state belongs to orchestration tooling
  - durable dataset state belongs to USR
  - notifier delivery state belongs to notifier tooling
  - workload-domain artifacts belong to the producing tool
- Active shared USR dataset ids are flat owner-first names. Nested paths may not
  encode producer routing for live shared handoffs; use `root_kind`,
  `owner_tool`, overlays, and study metadata for provenance.
- Curated study-facing workspace and runbook examples should default USR sinks
  to an explicit shared USR root rather than an implicit workspace-local
  dataset mirror.
- Workspace-local export roots and external USR roots remain allowed only when
  the workflow makes that storage boundary explicit.
- Shared data-plane behaviors such as overlay compaction and part-management are expressed with USR semantics (`usr-overlay-*`) instead of tool-specific command names.
- Cross-dataset overlay carry must stay explicit and narrow: no implicit merge-side overlay copying, no non-`id` carry keys, and no reserved-namespace transfer hidden behind convenience defaults.
- No hidden path fallback is allowed for orchestration accumulation; when required paths are missing or invalid, commands fail with actionable errors.
- Transient operational working directories and generated artifact roots are never root-level repo paths; disposable working state uses `/scratch`, while durable orchestration state remains workspace-scoped.

## Tool/package boundaries
- Tool-local behavior belongs under `src/dnadesign/<tool>/`.
- Shared top-level `src/dnadesign/utils` is disallowed; reusable helpers must either live inside a tool boundary or move into an explicitly versioned shared package.
- Shared cross-tool artifact schemas may live under `src/dnadesign/contracts/` when they are neutral, versioned, and used through public imports instead of sibling `src.*` internals.
- Shared test fixtures may live under `src/dnadesign/devtools/tests/support/`, but production code must not import them.
- Shared logic belongs in shared modules (`src/dnadesign/devtools/` or dedicated shared packages), not by copying scripts across tools.
- Cross-tool coupling should happen via documented artifacts/contracts (files, events, CLI contracts) or explicit public package APIs.
- Internal `dnadesign.<tool>.src.*` imports across tool boundaries are non-contractual and disallowed.

## Documentation model
- Root system-of-record docs define durable contracts and navigation.
- Deep procedures may live in top-level `docs/` or in the boundary-owning tool's docs when that tool owns the durable handoff contract.
- Root docs must stay abstract and route to one authoritative deep procedure for each cross-tool workflow.
- `docs/runbooks/README.md` is the centralized inventory surface for authoritative procedures; it must summarize owner-local runbooks and workflows without becoming a duplicate procedure tree.
- Inventory rows in `docs/runbooks/README.md` must mirror `Registry-id`, `Type`, `Plane`, `Execution-kind`, `Status-kind`, and `Summary` declared in the linked owner-local procedure; drift is a docs-check failure.
- Tool top-level READMEs stay lightweight: one narrative paragraph plus quick links, with the tool-local docs index listed first.
- Keep indexes short and link outward; avoid duplicating long procedures in multiple places or repeating adjacent links to the same file.
- Terminology is controlled:
  - `route`: index entry only
  - `runbook`: authoritative operator procedure with ordered commands and verification
  - `workflow`: downstream tool-owned branch or state-machine procedure when the tool intentionally uses a `workflows/` subtree
  - `tutorial`: pedagogical walkthrough, not the authority contract
  - `demo`: packaged sample assets or tracer-bullet profiles, not the authority contract
- Terminology and workflow ids are strict contracts, not compatibility layers:
  - do not keep deprecated aliases alive in authoritative surfaces
  - when an identifier is retired, reject it explicitly and point to the supported replacement
- Cross-tool route/runbook/workflow docs must declare:
  - `Type`
  - `Plane`
  - `Owner-boundary`
  - `Entry artifact`
  - `Exit artifact`
- `registry` must always be domain-qualified in docs and code-facing prose (for example `USR namespace registry`, `construct workspace registry`, `OPAL plugin registry`).
- `docs/operations/` is the control-plane orchestration surface; durable USR-backed data-plane procedures belong under `src/dnadesign/usr/docs/operations/`.

## References
- Architecture map: `ARCHITECTURE.md`
- Security policy: `SECURITY.md`
- Reliability/operations model: `RELIABILITY.md`
- Docs index: `docs/README.md`
- Maintainer docs: `docs/dev/README.md`
