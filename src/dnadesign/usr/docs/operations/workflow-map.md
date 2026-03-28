# USR Workflow Map

**Type:** route
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** operator intent for a USR-backed workflow branch
**Exit artifact:** chosen runbook link plus summary command chain

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16


Use this page to pick a command chain quickly, then open the linked runbook for full detail. The command blocks below are short summaries, not full procedures.

## Context preamble

Set shared roots once before you copy any branch-specific summary fragment.
Only keep the variables that the chosen branch actually uses, then open the linked runbook for exact value derivation and verification detail.

```bash
# Choose one operator workspace or scratch root for machine-readable artifacts.
# Set the operator workspace root that will own audit artifacts for this run.
WORKFLOW_ROOT="${WORKFLOW_ROOT:-$PWD}"
# Set the workspace-scoped artifact directory used by the summary fragments below.
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$WORKFLOW_ROOT/outputs/logs/usr-workflow-map}"
# Create the artifact directory before any machine-readable reports are emitted.
mkdir -p "$ARTIFACT_ROOT"

# Shared branch inputs. Keep only the ones needed by the branch you are running.
# Set the dataset id used by the sync-oriented branches.
DATASET_ID="my_dataset"
# Set the local USR root used by infer or notify dry-run branches.
LOCAL_USR_ROOT="<local-usr-root>"
# Set the USR root used by construct-backed branches.
USR_ROOT="$LOCAL_USR_ROOT"
# Set the construct or infer workspace root used by workspace-backed branches.
WORKSPACE_ROOT="<workspace-root>"
# Set the first infer config used by the feature-matrix branch.
INFER_CONFIG_7B="<path-to-infer-config.yaml>"
# Set the OPAL campaign workdir used by the active-learning branch.
OPAL_WORKDIR="<path-to-opal-campaign-dir>"
```

## Bootstrap from remote -> local clone

When HPC already has dataset contents and local does not.

```bash
# Set the dataset id used across sync calls.
DATASET_ID="my_dataset"
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
DATASET_ID="my_dataset"
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
DATASET_ID="my_dataset"
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
DATASET_ID="my_dataset"
# Pull the latest dataset state from HPC.
uv run usr pull "$DATASET_ID" bu-scc -y
# Run infer against the USR dataset and write derived outputs back into the dataset.
uv run infer run --preset evo2/extract_logits_ll --usr "$DATASET_ID" --usr-root "$LOCAL_USR_ROOT" --field sequence --device cpu --write-back
# Push derived outputs back to HPC.
uv run usr push "$DATASET_ID" bu-scc -y
```

Details: [chained-densegen-infer-sync-runbook.md](chained-densegen-infer-sync-runbook.md)

## Multi-source USR assembly -> Construct -> Infer

Use this when upstream records already live in multiple USR datasets and one downstream construct dataset should remain the durable infer/notify handoff boundary.

```bash
# Choose one existing USR dataset to mutate as the merged construct-input dataset.
PRIMARY_INPUT_DATASET="promoter_sources_control"
# Choose one additional upstream source dataset to fold into that construct-input dataset.
EXTRA_INPUT_DATASET="promoter_sources_densegen"
# Choose one downstream construct/infer handoff dataset id.
DOWNSTREAM_DATASET="multi_source_construct_truth_demo"
# Validate both upstream source datasets before the merge step.
uv run usr --root "$USR_ROOT" validate "$PRIMARY_INPUT_DATASET" --strict # Validate the primary construct-input dataset before it is mutated.
uv run usr --root "$USR_ROOT" validate "$EXTRA_INPUT_DATASET" --strict # Validate the extra upstream source dataset before it is merged.
# Merge the extra source into the primary input dataset and carry source labels explicitly.
uv run usr --root "$USR_ROOT" maintenance merge --dest "$PRIMARY_INPUT_DATASET" --src "$EXTRA_INPUT_DATASET" --union-columns --if-duplicate error --carry-namespace usr_label # Merge rows into the primary construct-input dataset and carry `usr_label`.
# Validate each construct project against the merged input dataset and shared downstream dataset.
uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_a_window --runtime # Validate slot_a against the merged input dataset and shared downstream dataset.
uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_b_window --runtime # Validate slot_b against the merged input dataset and shared downstream dataset.
# Materialize both construct projects after validation succeeds.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_a_window # Materialize slot_a rows into the shared downstream dataset.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_b_window # Materialize slot_b rows into the shared downstream dataset.
# Dry-run infer against the same downstream dataset after the shared continuation creates the infer config.
uv run infer run --config "$WORKSPACE_ROOT/infer.construct-shared-dataset.yaml" --dry-run # Dry-run infer against the same downstream dataset.
# Dry-run downstream event consumption against the same downstream dataset.
uv run notify usr-events watch --events "$USR_ROOT/$DOWNSTREAM_DATASET/.events.log" --provider generic --dry-run --no-advance-cursor-on-dry-run # Dry-run downstream event watching against the same downstream dataset.
```

Details: [multi-source-shared-dataset-assembly.md](multi-source-shared-dataset-assembly.md)

## Construct -> USR -> Infer shared dataset loop

Use this when construct should consolidate explicit source/template pairs into one USR-backed dataset before infer adds namespaced overlays.

```bash
# Create or reuse one semantic downstream dataset id.
DATASET_ID="anchor_template_shared_dataset_demo"
# Validate both construct projects that feed the shared dataset.
uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_a_window --runtime # Validate slot_a runtime roots and output contract.
uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_b_window --runtime # Validate slot_b runtime roots and output contract.
# Materialize both construct projects after validation succeeds.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_a_window # Materialize slot_a rows into the shared dataset.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_b_window # Materialize slot_b rows into the shared dataset.
# Dry-run infer against the construct dataset before any model execution.
uv run infer run --config "$WORKSPACE_ROOT/infer.construct-shared-dataset.yaml" --dry-run
# Dry-run downstream event consumption against the same dataset.
uv run notify usr-events watch --events "$USR_ROOT/$DATASET_ID/.events.log" --provider generic --dry-run --no-advance-cursor-on-dry-run
```

Details: [construct-infer-shared-dataset-runbook.md](construct-infer-shared-dataset-runbook.md)

## Promoter feature matrix -> Cluster or OPAL prep

Use this when DenseGen anchors, wildtype/manual promoters, and optional construct-expanded contexts should all feed one infer-annotated dataset before downstream clustering or active learning begins.

```bash
# Reuse one merged source dataset or one construct-expanded downstream dataset as the infer target plane.
FEATURE_DATASET="promoter_feature_matrix_demo"
# Dry-run the first infer matrix lane before model execution.
uv run infer run --config "$INFER_CONFIG_7B" --dry-run
# Execute the selected infer matrix lane and write namespaced feature columns back to the dataset.
uv run infer run --config "$INFER_CONFIG_7B"
# Explore one explicit infer-derived vector column with cluster.
uv run cluster fit --dataset "$FEATURE_DATASET" --x-col infer__evo2_7b__anchor_only_7b_features__intermediate_embedding__block26_mlp_out__seq_mean --name promoter_matrix_clusters_v1 --write --allow-overwrite
# Hand the same dataset plus the chosen X column into OPAL after campaign.yaml points at that dataset.
uv run opal validate -c "$OPAL_WORKDIR/configs/campaign.yaml" # Validate the USR-backed OPAL campaign before any rounds run.
uv run opal run -c "$OPAL_WORKDIR/configs/campaign.yaml" --labels-as-of 0 # Train, score, and select against the chosen infer-derived X column.
```

Details: [promoter-characterization-feature-matrix.md](promoter-characterization-feature-matrix.md)

## Machine-readable sync decisions

Use this when command chains are orchestrated by scripts, notebooks, or higher-level tools.

```bash
# Set the dataset id used across sync calls.
DATASET_ID="my_dataset"
# Emit machine-readable sync decision artifact.
uv run usr diff "$DATASET_ID" bu-scc --audit-json-out "$ARTIFACT_ROOT/usr-sync-audit.json"
# Read the high-level decision payload for orchestration logic.
jq -r '.data | {action, transfer_state, primary_changed: .primary.changed, derived_changed: ._derived.changed, aux_changed: ._auxiliary.changed}' "$ARTIFACT_ROOT/usr-sync-audit.json"
# Read exact sidecar file deltas for transfer decisions.
jq -r '.data | {derived_local_only: ._derived.local_only, derived_remote_only: ._derived.remote_only, aux_local_only: ._auxiliary.local_only, aux_remote_only: ._auxiliary.remote_only}' "$ARTIFACT_ROOT/usr-sync-audit.json"
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
USR_HARNESS_REPORT_PATH="$ARTIFACT_ROOT/usr-harness-report.json" \
USR_HARNESS_SYNC_AUDIT_REPORT_PATH="$ARTIFACT_ROOT/usr-sync-audit-drill-report.json" \
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
USR_HARNESS_REPORT_PATH="$ARTIFACT_ROOT/usr-harness-report.json" \
  bash src/dnadesign/usr/scripts/run_usr_harness_cycle.sh
# Optional: include the local sync audit drill in the harness cycle.
USR_HARNESS_RUN_SYNC_AUDIT_DRILL=1 \
USR_HARNESS_SYNC_AUDIT_REPORT_PATH="$ARTIFACT_ROOT/usr-sync-audit-drill-report.json" \
  bash src/dnadesign/usr/scripts/run_usr_harness_cycle.sh
```

## Deterministic sync audit drill

Use this when you want an end-to-end `diff/pull/push` drill with machine-readable audit artifacts across `_derived`, `_auxiliary`, and `_registry` perturbations.

```bash
# Run the local sync audit drill with an explicit report path.
uv run python src/dnadesign/usr/scripts/run_usr_sync_audit_drill.py \
  --report-json "$ARTIFACT_ROOT/usr-sync-audit-drill-report.json"
# Optional: keep local and remote drill roots for manual inspection.
uv run python src/dnadesign/usr/scripts/run_usr_sync_audit_drill.py \
  --work-dir "$ARTIFACT_ROOT/usr-sync-audit-drill" \
  --report-json "$ARTIFACT_ROOT/usr-sync-audit-drill-report.json"
```
