# Start Here

Use this page as a lightweight entrypoint before diving into full references.

## Workflow-first navigation

- Canonical docs index: [docs/README.md](README.md)
- USR task-first workflow map: [src/dnadesign/usr/docs/operations/workflow-map.md](../src/dnadesign/usr/docs/operations/workflow-map.md)
- USR sync command contract: [src/dnadesign/usr/docs/operations/sync.md](../src/dnadesign/usr/docs/operations/sync.md)
- USR sync audit loop: [src/dnadesign/usr/docs/operations/sync-audit-loop.md](../src/dnadesign/usr/docs/operations/sync-audit-loop.md)
- Iterative HPC sync loop: [src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md)
- Chained DenseGen -> USR -> Infer workflow: [src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md)
- Adversarial sync fidelity drills: [src/dnadesign/usr/docs/operations/sync-fidelity-drills.md](../src/dnadesign/usr/docs/operations/sync-fidelity-drills.md)

## Decision ladders

- Need iterative HPC sync for large datasets:
  1. [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md)
  2. [USR HPC sync flow](../src/dnadesign/usr/docs/operations/hpc-agent-sync-flow.md)
  3. [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md)
- Need DenseGen -> USR -> Infer chained updates:
  1. [USR workflow map](../src/dnadesign/usr/docs/operations/workflow-map.md)
  2. [USR chained DenseGen+Infer demo](../src/dnadesign/usr/docs/operations/chained-densegen-infer-sync-demo.md)
  3. [Infer README](../src/dnadesign/infer/README.md)
- Need to diagnose sidecar or overlay fidelity mismatches:
  1. [USR sync command contract](../src/dnadesign/usr/docs/operations/sync.md)
  2. [USR sync fidelity drills](../src/dnadesign/usr/docs/operations/sync-fidelity-drills.md)
  3. [USR sync audit loop](../src/dnadesign/usr/docs/operations/sync-audit-loop.md)

## Task shortcuts

- Need to bootstrap a repo clone and environment: [docs/installation.md](installation.md)
- Need BU SCC submit/run patterns: [docs/bu-scc/README.md](bu-scc/README.md)
- Need webhook and event-operator workflows: [docs/notify/README.md](notify/README.md)
- Need architecture and reliability boundaries: [ARCHITECTURE.md](../ARCHITECTURE.md), [RELIABILITY.md](../RELIABILITY.md)
