# USR operations runbooks

This directory holds operator runbooks in progressive order.

## Read order

1. [sync.md](sync.md): sync command contract and troubleshooting.
2. [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md): preflight/run/verify loop for iterative HPC pulls.
3. [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md): chained DenseGen and Infer workflow with bidirectional sync.

## Operator contract

- Run `diff` before transfer decisions.
- Default dataset sync contract is `--verify hash` with strict sidecar checks enabled.
- Use `--no-verify-sidecars` only when an operator explicitly accepts reduced fidelity checks.
- Use sync audit output (`Primary`, `.events.log`, `_snapshots`, `_derived`) to decide pull/push actions.
