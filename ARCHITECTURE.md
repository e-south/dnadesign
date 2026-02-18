# ARCHITECTURE

**Type:** system-of-record
**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

## At a glance
`dnadesign` is a uv-managed monorepo of modular bioinformatics tools under `src/dnadesign/`, with shared CI/devtools and operator runbooks in `docs/`.
This file is the architecture map: it names system boundaries, major flows, and invariants, then links to deeper operational/reference docs.

## Contents
- [Repository shape](#repository-shape)
- [System boundaries](#system-boundaries)
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
- Shared developer infrastructure: devtools modules provide CI scope detection, docs checks, coverage gates, and quality entropy reporting.
- Operator-facing HPC boundary: BU SCC guidance and templates live in `docs/bu-scc/`.

## High-level data flows
- DenseGen/other producers -> USR event stream (`.events.log`) -> Notify watcher/webhook sink.
- Tool outputs -> dataset artifacts (for example Parquet tables) -> downstream analysis tools (cluster, billboard, nmf, latdna, cruncher, tfkdanalysis).
- Developer workflow -> fast CI lane (lint/docs/fast tests + coverage gate) and heavy FIMO/integration lane when required.

## Architecture invariants
- No silent fallbacks: missing required inputs/dependencies must fail fast with actionable errors.
- Boundary contracts are explicit: CI scope, tool coverage baselines, and marker-based heavy tests are enforced in code.
- Docs use progressive disclosure: root docs are maps; deep procedures stay in runbooks/reference docs.

## Where to go deeper
- Maintainer index: `docs/dev/README.md`
- Monorepo organization audit: `docs/dev/monorepo-organization-audit.md`
- BU SCC operational source of truth: `docs/bu-scc/sge-hpc-ops/SKILL.md`
- Notify event contract: `docs/notify/usr-events.md`
- Reliability operations: `RELIABILITY.md`
- Security policy and secrets handling: `SECURITY.md`
- Engineering invariants: `DESIGN.md`
- Active/proposed work tracking: `PLANS.md`
