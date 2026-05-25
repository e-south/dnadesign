# USR operations runbooks

**Type:** route
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** USR dataset, sync, or cross-tool shared-dataset work that still needs a route
**Exit artifact:** USR runbook or downstream handoff route

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-24

Use this index to pick the next dataset, sync, or handoff step.

## Sync and transfer

| Need | Runbook |
| --- | --- |
| Pick a command chain by scenario, then drill into exact steps | [workflow-map.md](routes/workflow-map.md) |
| Choose sync quickstart, setup, modes, or troubleshooting | [sync.md](sync/README.md) |
| Run the minimum daily `diff` -> `pull` -> `push` loop | [sync-quickstart.md](sync/quickstart.md) |
| Configure one-time SSH keys, remote profiles, and rotation | [sync-setup.md](sync/setup.md) |
| Map source and target paths for dataset-directory vs file mode | [sync-modes.md](sync/modes.md) |
| Diagnose transfer and verification failures in deterministic order | [sync-troubleshooting.md](sync/troubleshooting.md) |
| Emit machine-readable transfer decisions for chained commands | [sync-audit-loop.md](sync/audit-loop.md) |
| Iterate HPC batch writes with local pull/verify checkpoints | [hpc-agent-sync-flow.md](sync/hpc-agent-flow.md) |
| Chain DenseGen and Infer updates with bidirectional sync | [chained-densegen-infer-sync-runbook.md](sync/chained-densegen-infer-runbook.md) |

## Shared dataset assembly

| Need | Runbook |
| --- | --- |
| Combine multiple USR-backed producer datasets before construct and infer share one downstream dataset | [multi-source-shared-dataset-assembly.md](assembly/multi-source-shared-dataset.md) |
| Build one construct-backed dataset, then hand it to infer | [construct-infer-shared-dataset-runbook.md](assembly/construct-infer-shared-dataset-runbook.md) |
| Route Permuter RT-lnRNA (`rt_lnrna`) variants through study-owned construct-subject promotion, Construct context realization, and Infer sidecars | [permuter-construct-infer-shared-dataset.md](assembly/permuter-construct-infer-shared-dataset.md) |

## Promoter study workflows

| Need | Runbook |
| --- | --- |
| Bootstrap a fresh thread, recover a missing study record, or find the next study-owned handoff | [../../../../../docs/studies/README.md](../../../../../docs/studies/README.md) |
| Current study status | [status contract](../../../../../docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md) |
| Current study preflight | [preflight contract](../../../../../docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md) |
| Build one infer-annotated promoter-study feature dataset, then continue to cluster or prepare OPAL after choosing one explicit `X` column | [promoter-characterization-feature-matrix.md](promoter/characterization-feature-matrix.md) |
| Review the full DenseGen/manual/wildtype -> optional Construct -> Infer Evo2 -> Notify/Cluster/OPAL path before choosing one branch | [promoter-evo2-journey.md](promoter/evo2-journey.md) |

## Validation and drills

| Need | Runbook |
| --- | --- |
| Pressure-test schema, sidecar, and overlay fidelity failure paths | [sync-fidelity-drills.md](sync/fidelity-drills.md) |
| Run one-pass mock batch plus adversarial pressure checks | [workflow-map.md](routes/workflow-map.md#pressure-test-loop-mock-batch--adversarial-schemas) |
| Run deterministic preflight/run/verify cycle (optional drill toggle) | [workflow-map.md](routes/workflow-map.md#deterministic-harness-cycle) |
| Run deterministic local `diff`/`pull`/`push` audit drill | [workflow-map.md](routes/workflow-map.md#deterministic-sync-audit-drill) using `uv run usr-sync-audit-drill` |

## Read order

1. [workflow-map.md](routes/workflow-map.md): command chains by task.
2. Sync lifecycle:
   [sync-quickstart.md](sync/quickstart.md),
   [sync-setup.md](sync/setup.md),
   [sync-modes.md](sync/modes.md),
   [sync-troubleshooting.md](sync/troubleshooting.md),
   [sync-audit-loop.md](sync/audit-loop.md),
   [hpc-agent-sync-flow.md](sync/hpc-agent-flow.md),
   [chained-densegen-infer-sync-runbook.md](sync/chained-densegen-infer-runbook.md)
3. Shared dataset assembly:
   [multi-source-shared-dataset-assembly.md](assembly/multi-source-shared-dataset.md),
   [construct-infer-shared-dataset-runbook.md](assembly/construct-infer-shared-dataset-runbook.md),
   [permuter-construct-infer-shared-dataset.md](assembly/permuter-construct-infer-shared-dataset.md)
4. Promoter study workflows:
   [promoter-evo2-journey.md](promoter/evo2-journey.md),
   [status contract](../../../../../docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/status.md),
   [preflight contract](../../../../../docs/studies/stress_ethanol_cipro_growth/operations/catalog/contracts/preflight.md),
   [promoter-characterization-feature-matrix.md](promoter/characterization-feature-matrix.md)
5. Validation drills:
   [sync-fidelity-drills.md](sync/fidelity-drills.md),
   [workflow-map.md#deterministic-sync-audit-drill](routes/workflow-map.md#deterministic-sync-audit-drill) using `uv run usr-sync-audit-drill`

## Operator contract

- Run `diff` before transfer decisions.
- Default dataset sync contract is `--verify hash` with strict sidecar and `_derived`/`_auxiliary` content-hash checks enabled.
- Use `--no-verify-sidecars` only when an operator explicitly accepts reduced fidelity checks.
- Use `--no-verify-derived-hashes` only when an operator explicitly accepts reduced content-hash fidelity.
- Use sync audit output (`Primary`, `.events.log`, `_snapshots`, `_derived`) to decide pull/push actions.
