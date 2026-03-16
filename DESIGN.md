# DESIGN

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-03

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
- DenseGen run-mode behavior stays explicit and fail-fast:
  - `--mode auto` must resolve from workspace state
  - `--mode fresh` is blocked when resume artifacts exist unless reset is explicitly acknowledged
- Cross-tool output ownership is orthogonal:
  - `ops`: orchestration logs/audits/runbooks
  - `notify`: profile/cursor/spool and webhook delivery
  - `usr`: dataset records and `.events.log`
  - `densegen` and `infer`: workload-domain artifacts and dataset updates
- Curated workspace and runbook examples must default USR dataset roots to `<workspace-root>/outputs/usr_datasets`.
- External USR roots remain allowed only when the workflow makes that storage boundary explicit.
- Shared data-plane behaviors such as overlay compaction and part-management are expressed with USR semantics (`usr-overlay-*`) instead of tool-specific command names.
- Cross-dataset overlay carry must stay explicit and narrow: no implicit merge-side overlay copying, no non-`id` carry keys, and no reserved-namespace transfer hidden behind convenience defaults.
- No hidden path fallback is allowed for orchestration accumulation; when required paths are missing or invalid, commands fail with actionable errors.
- Transient operational working directories are never root-level repo paths; disposable working state uses `/scratch`, while durable orchestration state remains workspace-scoped.

## Tool/package boundaries
- Tool-local behavior belongs under `src/dnadesign/<tool>/`.
- Shared top-level `src/dnadesign/utils` is disallowed; reusable helpers must either live inside a tool boundary or move into an explicitly versioned shared package.
- Shared logic belongs in shared modules (`src/dnadesign/devtools/` or dedicated shared packages), not by copying scripts across tools.
- Cross-tool coupling should happen via documented artifacts/contracts (files, events, CLI contracts) or explicit public package APIs.
- Internal `dnadesign.<tool>.src.*` imports across tool boundaries are non-contractual and disallowed.

## Documentation model
- Root system-of-record docs define durable contracts and navigation.
- Deep procedures may live in top-level `docs/` or in the boundary-owning tool's docs when that tool owns the durable handoff contract.
- Root docs must route to one authoritative deep procedure for each cross-tool workflow.
- Keep indexes short and link outward; avoid duplicating long procedures in multiple places.
- Terminology is controlled:
  - `route`: index entry only
  - `runbook`: authoritative operator procedure with ordered commands and verification
  - `workflow`: downstream tool-owned branch or state-machine procedure when the tool intentionally uses a `workflows/` subtree
  - `tutorial`: pedagogical walkthrough, not the authority contract
  - `demo`: packaged sample assets or tracer-bullet profiles, not the authority contract
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
