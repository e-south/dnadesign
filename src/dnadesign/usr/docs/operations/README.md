# USR operations runbooks

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This directory holds operator runbooks in progressive order.

## Task shortcuts

| Need | Runbook |
| --- | --- |
| Pick a command chain by scenario, then drill into exact steps | [workflow-map.md](workflow-map.md) |
| Open the sync contract router before choosing quickstart/setup/modes/troubleshooting | [sync.md](sync.md) |
| Run the minimum daily `diff` -> `pull` -> `push` loop | [sync-quickstart.md](sync-quickstart.md) |
| Configure one-time SSH keys, remote profiles, and rotation | [sync-setup.md](sync-setup.md) |
| Map source and target paths for dataset-directory vs file mode | [sync-modes.md](sync-modes.md) |
| Diagnose transfer and verification failures in deterministic order | [sync-troubleshooting.md](sync-troubleshooting.md) |
| Emit machine-readable transfer decisions for chained commands | [sync-audit-loop.md](sync-audit-loop.md) |
| Iterate HPC batch writes with local pull/verify checkpoints | [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md) |
| Chain DenseGen and Infer updates with bidirectional sync | [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md) |
| Pressure-test schema, sidecar, and overlay fidelity failure paths | [sync-fidelity-drills.md](sync-fidelity-drills.md) |
| Run one-pass mock batch plus adversarial pressure checks | [workflow-map.md](workflow-map.md#pressure-test-loop-mock-batch--adversarial-schemas) |
| Run deterministic preflight/run/verify cycle (optional drill toggle) | [workflow-map.md](workflow-map.md#deterministic-harness-cycle) |
| Run deterministic local `diff`/`pull`/`push` audit drill | [workflow-map.md](workflow-map.md#deterministic-sync-audit-drill) |

## Read order

1. [workflow-map.md](workflow-map.md): task-first command chains.
2. [sync-quickstart.md](sync-quickstart.md): baseline operator loop.
3. [sync-setup.md](sync-setup.md): key and remote setup contract.
4. [sync-modes.md](sync-modes.md): dataset/file mode mapping contract.
5. [sync-troubleshooting.md](sync-troubleshooting.md): failure diagnosis sequence.
6. [sync-audit-loop.md](sync-audit-loop.md): machine-readable sync audit loop for automation and notebook/tool chaining.
7. [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md): preflight/run/verify loop for iterative HPC pulls.
8. [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md): chained DenseGen and Infer workflow with bidirectional sync.
9. [sync-fidelity-drills.md](sync-fidelity-drills.md): adversarial drills for strict sidecar plus overlay fidelity.
10. [workflow-map.md#deterministic-sync-audit-drill](workflow-map.md#deterministic-sync-audit-drill): deterministic `run_usr_sync_audit_drill.py` command path with machine-readable audit output.

## Operator contract

- Run `diff` before transfer decisions.
- Default dataset sync contract is `--verify hash` with strict sidecar and `_derived`/`_auxiliary` content-hash checks enabled.
- Use `--no-verify-sidecars` only when an operator explicitly accepts reduced fidelity checks.
- Use `--no-verify-derived-hashes` only when an operator explicitly accepts reduced content-hash fidelity.
- Use sync audit output (`Primary`, `.events.log`, `_snapshots`, `_derived`) to decide pull/push actions.
