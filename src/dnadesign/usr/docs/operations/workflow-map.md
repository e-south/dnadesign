# USR Workflow Map

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Use this page to pick a command chain quickly, then open the linked runbook for full detail.

## Bootstrap from remote -> local clone

When HPC already has dataset contents and local does not.

```bash
# Set the dataset id used across sync calls.
DATASET_ID="densegen/my_dataset"
# Preview divergence before transfer.
uv run usr diff "$DATASET_ID" bu-scc
# Pull remote dataset into local root.
uv run usr pull "$DATASET_ID" bu-scc -y
# Confirm no remaining remote deltas.
uv run usr diff "$DATASET_ID" bu-scc
```

Details: [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md#bootstrap-from-either-side)

## Bootstrap from local -> remote clone

When local already has dataset contents and HPC does not.

```bash
# Set the dataset id used across sync calls.
DATASET_ID="densegen/my_dataset"
# Preview divergence before transfer.
uv run usr diff "$DATASET_ID" bu-scc
# Push local dataset into remote root.
uv run usr push "$DATASET_ID" bu-scc -y
# Confirm no remaining remote deltas.
uv run usr diff "$DATASET_ID" bu-scc
```

Details: [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md#bootstrap-from-either-side)

## Iterative HPC batch loop

Use this for repeated remote writes and local analysis refresh.

```bash
# Set the dataset id used across sync calls.
DATASET_ID="densegen/my_dataset"
# Preview divergence before transfer.
uv run usr diff "$DATASET_ID" bu-scc
# Pull remote updates into local root.
uv run usr pull "$DATASET_ID" bu-scc -y
# run local analysis/notebook against local USR root
# Confirm no remaining remote deltas.
uv run usr diff "$DATASET_ID" bu-scc
```

Details: [hpc-agent-sync-flow.md](hpc-agent-sync-flow.md#run-loop-hpc-side-writes-local-side-reads)

## DenseGen -> USR -> Infer -> USR chained loop

Use this when DenseGen runs on HPC and Infer annotations are produced locally or on another host.

```bash
# Set the dataset id used across sync calls.
DATASET_ID="densegen/my_dataset"
# Pull the latest dataset state from HPC.
uv run usr pull "$DATASET_ID" bu-scc -y
# Run infer against the USR dataset and write derived outputs.
uv run infer run --preset evo2/extract_logits_ll --usr "$DATASET_ID" --root "$LOCAL_USR_ROOT"
# Push derived outputs back to HPC.
uv run usr push "$DATASET_ID" bu-scc -y
```

Details: [chained-densegen-infer-sync-demo.md](chained-densegen-infer-sync-demo.md)

## Machine-readable sync decisions

Use this when command chains are orchestrated by scripts, notebooks, or higher-level tools.

```bash
# Set the dataset id used across sync calls.
DATASET_ID="densegen/my_dataset"
# Emit machine-readable sync decision artifact.
uv run usr diff "$DATASET_ID" bu-scc --audit-json-out /tmp/usr-sync-audit.json
# Read the diff decision payload for orchestration logic.
jq -r '.changes' /tmp/usr-sync-audit.json
# Read exact sidecar file deltas for transfer decisions.
jq -r '.data | {derived_local_only: ._derived.local_only, derived_remote_only: ._derived.remote_only, aux_local_only: ._auxiliary.local_only, aux_remote_only: ._auxiliary.remote_only}' /tmp/usr-sync-audit.json
```

Details: [sync-audit-loop.md](sync-audit-loop.md)

## Failure drills and contract checks

- Sidecar and hash-fidelity drills: [sync-fidelity-drills.md](sync-fidelity-drills.md)
- Full command contract and option semantics: [sync.md](sync.md)

## Pressure-test loop (mock batch + adversarial schemas)

Use this before or after sync/overlay refactors to validate iterative transfer behavior and schema hardening in one pass.

```bash
# Run deterministic harness cycle with optional sync-audit drill enabled.
USR_HARNESS_RUN_SYNC_AUDIT_DRILL=1 \
USR_HARNESS_REPORT_PATH=/tmp/usr-harness-report.json \
USR_HARNESS_SYNC_AUDIT_REPORT_PATH=/tmp/usr-sync-audit-drill-report.json \
  bash src/dnadesign/usr/scripts/run_usr_harness_cycle.sh

# Re-run targeted adversarial suites directly when iterating quickly.
uv run pytest -q \
  src/dnadesign/usr/tests/test_sync_iterative_batch_flow.py \
  src/dnadesign/usr/tests/test_sync_schema_adversarial.py \
  src/dnadesign/usr/tests/test_usr_sync_audit_drill_script.py
```

Details: [sync-fidelity-drills.md](sync-fidelity-drills.md), [sync-audit-loop.md](sync-audit-loop.md)

## Deterministic harness cycle

Use this when you want one reproducible preflight -> run -> verify pass before or after refactors.

```bash
# Run the deterministic USR harness cycle from repo root.
bash src/dnadesign/usr/scripts/run_usr_harness_cycle.sh
# Optional: emit machine-readable harness evidence.
USR_HARNESS_REPORT_PATH=/tmp/usr-harness-report.json \
  bash src/dnadesign/usr/scripts/run_usr_harness_cycle.sh
# Optional: include the local sync audit drill in the harness cycle.
USR_HARNESS_RUN_SYNC_AUDIT_DRILL=1 \
USR_HARNESS_SYNC_AUDIT_REPORT_PATH=/tmp/usr-sync-audit-drill-report.json \
  bash src/dnadesign/usr/scripts/run_usr_harness_cycle.sh
```

## Deterministic sync audit drill

Use this when you want an end-to-end `diff/pull/push` drill with machine-readable audit artifacts across `_derived`, `_auxiliary`, and `_registry` perturbations.

```bash
# Run the local sync audit drill with an explicit report path.
uv run python src/dnadesign/usr/scripts/run_usr_sync_audit_drill.py \
  --report-json /tmp/usr-sync-audit-drill-report.json
# Optional: keep local and remote drill roots for manual inspection.
uv run python src/dnadesign/usr/scripts/run_usr_sync_audit_drill.py \
  --work-dir /tmp/usr-sync-audit-drill \
  --report-json /tmp/usr-sync-audit-drill-report.json
```
