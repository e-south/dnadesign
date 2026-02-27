# USR Workflow Map

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
```

Details: [sync-audit-loop.md](sync-audit-loop.md)

## Failure drills and contract checks

- Sidecar and hash-fidelity drills: [sync-fidelity-drills.md](sync-fidelity-drills.md)
- Full command contract and option semantics: [sync.md](sync.md)
