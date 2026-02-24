# DESIGN

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

## At a glance
This document defines repo-wide engineering principles, invariants, and boundary rules.
Use it when adding or changing behavior so tools remain decoupled, assertive, and easy to evolve.

## Contents
- [Core principles](#core-principles)
- [Boundary contracts](#boundary-contracts)
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

## Tool/package boundaries
- Tool-local behavior belongs under `src/dnadesign/<tool>/`.
- Shared logic belongs in shared modules (`src/dnadesign/devtools/` or dedicated shared packages), not by copying scripts across tools.
- Cross-tool coupling should happen via documented artifacts/contracts (files, events, CLI contracts) or explicit public package APIs.
- Internal `dnadesign.<tool>.src.*` imports across tool boundaries are non-contractual and disallowed.

## Documentation model
- Root system-of-record docs define durable contracts and navigation.
- `docs/` holds runbooks, references, decisions, quality guidance, and execution plans.
- Keep indexes short and link outward; avoid duplicating long procedures in multiple places.

## References
- Architecture map: `ARCHITECTURE.md`
- Security policy: `SECURITY.md`
- Reliability/operations model: `RELIABILITY.md`
- Docs index: `docs/README.md`
- Maintainer docs: `docs/dev/README.md`
