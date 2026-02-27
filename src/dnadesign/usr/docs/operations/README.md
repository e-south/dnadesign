# USR operations runbooks

This directory holds operator runbooks in progressive order.

## Task shortcuts

| Need | Runbook |
| --- | --- |
| Pick a command chain quickly, then drill into detailed runbooks | [workflow-map.md](workflow-map.md) |
| Start with sync command contracts and failure semantics | [sync.md](sync.md) |
| Emit machine-readable transfer decisions for chained workflows | [sync-audit-loop.md](sync-audit-loop.md) |
| Iterate HPC batch writes with local pull/verify loops | [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md) |
| Chain DenseGen and Infer updates with bidirectional sync | [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md) |
| Pressure test schema/fidelity failure paths | [sync-fidelity-drills.md](sync-fidelity-drills.md) |

## Read order

1. [workflow-map.md](workflow-map.md): task-first command chains.
2. [sync.md](sync.md): sync command contract and troubleshooting.
3. [sync-audit-loop.md](sync-audit-loop.md): machine-readable sync audit loop for automation and notebook/tool chaining.
4. [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md): preflight/run/verify loop for iterative HPC pulls.
5. [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md): chained DenseGen and Infer workflow with bidirectional sync.
6. [sync-fidelity-drills.md](sync-fidelity-drills.md): adversarial drills for strict sidecar plus overlay fidelity.

## Operator contract

- Run `diff` before transfer decisions.
- Default dataset sync contract is `--verify hash` with strict sidecar and `_derived`/`_auxiliary` content-hash checks enabled.
- Use `--no-verify-sidecars` only when an operator explicitly accepts reduced fidelity checks.
- Use `--no-verify-derived-hashes` only when an operator explicitly accepts reduced content-hash fidelity.
- Use sync audit output (`Primary`, `.events.log`, `_snapshots`, `_derived`) to decide pull/push actions.
