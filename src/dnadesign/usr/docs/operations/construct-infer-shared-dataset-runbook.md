## Construct -> USR -> Infer Shared Dataset Runbook

**Type:** runbook
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** construct workspace plus one shared USR dataset target
**Exit artifact:** one infer-ready USR dataset plus downstream event stream
**Registry-id:** usr.data-plane.construct-infer-source-of-truth
**Summary:** Realize construct outputs into one shared USR dataset and use that dataset as the durable Infer handoff.
**Execution-kind:** staged
**Status-kind:** usr-dataset-state

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Use this runbook when construct should write one shared USR dataset and infer plus Notify should read that same dataset next.

If upstream inputs still span multiple USR datasets, start with [multi-source-shared-dataset-assembly.md](multi-source-shared-dataset-assembly.md) first, then return here once one construct-input dataset is already consolidated.

This runbook uses the packaged construct workspace as a local tracer bullet.
Its early commands intentionally write into that packaged workspace-local USR
root. For live promoter-study status and continuation, keep the study record's
declared shared USR root authoritative and repoint construct configs
deliberately before treating the dataset as the cross-tool source of truth.

For the live `stress_ethanol_cipro_growth` study, the real source datasets are
not the packaged demo inputs. Merge `densegen/study_stress_ethanol_cipro` and
`mg1655_promoters` in USR first, then point Construct at that merged source
dataset while keeping `plasmids` as the pDual-backed template dataset.

### Boundary decisions

- USR dataset roots are the durable data boundary, not git or workspace metadata.
- Root precedence in this flow is explicit config root first, then documented tool-specific env roots where supported; no packaged-dataset fallback is part of the contract.
- `construct` owns template realization and `construct__*` lineage.
- `infer` owns `infer__*` derived namespaces added after construct writes.
- `notify` consumes the consolidated dataset `.events.log`.
- `construct` remains one-input-selection plus one-template per job; broader matrices stay explicit as multiple workspace projects.

### When to use this path

- multiple canonical source records should end up in one semantic downstream dataset
- each realized sequence must retain construct input/template provenance
- infer should enrich the same dataset instead of creating a disconnected copy
- downstream tooling should read one dataset plus one event stream

### Design stance

- If upstream records live in multiple USR datasets and must be unified before construct, do that explicitly with `uv run usr maintenance merge ...`; do not hide multi-source consolidation inside one construct config.
- Use `uv run usr maintenance merge ... --carry-namespace <namespace>` when one compact, `id`-keyed overlay namespace such as `usr_label` must survive consolidation onto rows that actually survive the merge.
- Plain `uv run usr maintenance merge ...` still rewrites canonical base rows only. For namespaces that are not `id`-keyed or not yet compact, materialize or reattach them explicitly instead of expecting implicit carry-through.
- After Construct, use `uv run usr maintenance overlay-project ...` when authoritative upstream metadata must be reattached onto the realized downstream dataset through an explicit lineage key such as `construct__anchor_id`. This keeps Construct scoped to `construct__*` lineage and avoids widening it into a generic namespace pass-through tool.
- If multiple construct projects should accumulate into one semantic output dataset, keep each project auditable and point them at the same `output.target.dataset`.
- Start with `output.on_conflict=error` for fail-fast duplicate detection. Use `ignore` only for intentional idempotent reruns.

## Ordered procedure

### 1) Bootstrap a construct workspace and seed demo inputs

```bash
# Create a disposable root for this tracer-bullet workflow.
export WORK_ROOT="$(mktemp -d /tmp/construct-usr-infer-XXXXXX)"
# Pin the repo checkout so later file copies do not depend on the current shell directory.
export DNADESIGN_REPO_ROOT="$(git rev-parse --show-toplevel)"
# Scaffold the packaged shared-dataset construct workspace under that root.
uv run construct workspace init --id shared_dataset_demo --root "$WORK_ROOT" --profile anchor-template-shared-dataset-demo
# Reuse one workspace path across the remaining commands.
export WORKSPACE_ROOT="$WORK_ROOT/shared_dataset_demo"
# Seed packaged anchor and template records into the local tracer-bullet USR root.
uv run construct seed anchor-template-demo \
  --root "$WORKSPACE_ROOT/outputs/usr_datasets" \
  --manifest "$WORKSPACE_ROOT/inputs/seed_manifest.yaml"
```

### 2) Validate and materialize both packaged construct projects

```bash
# Keep one active construct config path explicit for downstream tooling.
export CONSTRUCT_CONFIG="$WORKSPACE_ROOT/config.slot_a.window.yaml"
# Verify workspace registry and config drift before execution.
uv run construct workspace doctor --workspace "$WORKSPACE_ROOT"
# Validate both packaged projects with resolved runtime roots.
uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_a_window --runtime # Validate slot_a runtime roots and template resolution.
uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_b_window --runtime # Validate slot_b runtime roots and template resolution.
# Dry-run both projects before mutating USR.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_a_window --dry-run # Preview slot_a writes without mutating USR state.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_b_window --dry-run # Preview slot_b writes without mutating USR state.
# Materialize the consolidated construct output dataset from both projects.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_a_window # Materialize slot_a rows into the shared dataset.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_b_window # Materialize slot_b rows into the shared dataset.
```

### 3) Verify the construct-backed shared dataset

```bash
# Reuse the local tracer-bullet USR root and one semantic dataset id.
export USR_ROOT="$WORKSPACE_ROOT/outputs/usr_datasets"
# Reuse the packaged construct output dataset id across verification and downstream tools.
export DATASET_ID="anchor_template_shared_dataset_demo"
# Confirm the written dataset satisfies the active USR registry.
uv run usr --root "$USR_ROOT" validate "$DATASET_ID" --strict
# Inspect human-readable source labels plus construct lineage needed by downstream tools.
uv run usr --root "$USR_ROOT" head "$DATASET_ID" -n 5 \
  --columns id,usr_label__primary,construct__input_dataset,construct__input_id,construct__template_id,construct__window_semantics
```

Expected outcome:

- records are canonical USR rows (`id`, `sequence`, `bio_type`, `length`, `source`, `created_at`)
- upstream source labels remain available in `usr_label__primary` / `usr_label__aliases`
- construct lineage remains attached in `construct__*` columns
- `.events.log` exists in `"$USR_ROOT/$DATASET_ID/.events.log"`

### 4) Scale the same pattern to more sources or templates

- represent each additional template or slot as another `construct.workspace.yaml` project
- point multiple projects at the same `output.target.dataset` only when one semantic shared dataset is intentional
- the packaged `anchor-template-shared-dataset-demo` profile is the turnkey two-project accumulation preset; use it as the tracer bullet before widening the matrix
- if multiple upstream USR datasets must be consolidated first, run `uv run usr maintenance merge ... --carry-namespace usr_label` when the upstream label overlay is compact and `id`-keyed; otherwise materialize or reattach the needed namespace explicitly before construct
- when a downstream dataset must recover authoritative upstream overlay metadata through a non-`id` lineage key, run `uv run usr maintenance overlay-project --src <source-dataset> --dest <downstream-dataset> --namespace <namespace> --src-join id --dest-join construct__anchor_id --allow-missing`
- rerun `construct workspace validate-project --runtime` for every project before `run-project`

### 5) Shared downstream continuation: prepare infer handoff against the construct dataset

Create a config dedicated to the construct output dataset. This keeps infer write-back explicit and repeatable. This section is also the shared downstream continuation used by the broader [multi-source-shared-dataset-assembly.md](multi-source-shared-dataset-assembly.md) runbook once one construct-backed downstream dataset already exists.

```bash
# Copy the packaged infer pressure-test config into the construct workspace.
cp "$DNADESIGN_REPO_ROOT/src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml" \
  "$WORKSPACE_ROOT/infer.construct-shared-dataset.yaml"
# Retarget the infer config to the construct output dataset and local CPU runtime.
uv run python - <<'PY'
import os
from pathlib import Path

import yaml

workspace_root = Path(os.environ["WORKSPACE_ROOT"])
config_path = workspace_root / "infer.construct-shared-dataset.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

config["jobs"][0]["ingest"]["dataset"] = os.environ["DATASET_ID"]
config["jobs"][0]["ingest"]["root"] = str(workspace_root / "outputs/usr_datasets")
config["model"]["device"] = "cpu"
config["model"]["precision"] = "fp32"
config["model"]["batch_size"] = 2

config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(config_path)
PY
# Validate infer config shape and the USR handoff contract.
uv run infer validate config --config "$WORKSPACE_ROOT/infer.construct-shared-dataset.yaml"
# Render the exact infer namespace registration required for write-back.
uv run infer validate usr-registry --config "$WORKSPACE_ROOT/infer.construct-shared-dataset.yaml"
# Dry-run infer against the construct dataset before any model execution.
uv run infer run --config "$WORKSPACE_ROOT/infer.construct-shared-dataset.yaml" --dry-run
```

Use the rendered `namespace register infer --columns ...` command before the first real infer write-back, then continue with the infer pressure-test paths:

- [Infer pressure-test runbook](../../../infer/docs/operations/pressure-test-agnostic-models.md)
- [Infer end-to-end demo](../../../infer/docs/tutorials/demo_pressure_test_usr_ops_notify.md)

### 6) Shared downstream continuation: verify downstream event consumption

```bash
# Resolve the construct-managed events path from the active config.
uv run notify setup resolve-events \
  --tool construct \
  --config "$CONSTRUCT_CONFIG" \
  --json
# Dry-run explicit event watching against the consolidated USR dataset.
uv run notify usr-events watch \
  --events "$USR_ROOT/$DATASET_ID/.events.log" \
  --provider generic \
  --dry-run \
  --no-advance-cursor-on-dry-run
# Optional workspace shorthand:
# export CONSTRUCT_WORKSPACE_ROOT="$WORK_ROOT"
# export DNADESIGN_REPO_ROOT=/abs/path/to/dnadesign
# uv run notify setup resolve-events --tool construct --workspace "shared_dataset_demo:slot_a_window" --json
# Resolve the infer-managed events path once the infer config exists.
uv run notify setup resolve-events \
  --tool infer \
  --config "$WORKSPACE_ROOT/infer.construct-shared-dataset.yaml" \
  --json
```

Resolver contract:

- `notify setup resolve-events --tool construct --config "$CONSTRUCT_CONFIG"` is supported
- `notify setup resolve-events --tool construct --workspace "<workspace-id>:<project-id>"` is supported when the workspace is discoverable from `src/dnadesign/construct/workspaces/` or `CONSTRUCT_WORKSPACE_ROOT`
- when using construct workspace shorthand from outside the repo checkout, set `DNADESIGN_REPO_ROOT=<repo-root>` alongside `CONSTRUCT_WORKSPACE_ROOT`, or pass `--config` explicitly
- explicit `--events "$USR_ROOT/$DATASET_ID/.events.log"` wiring remains valid when you already know the dataset path or want to bypass resolver mode entirely

## Verification checklist

- `construct workspace doctor` reports `workspace_doctor: ok`
- `construct workspace validate-project --runtime` resolves the intended roots and template
- `usr validate <dataset> --strict` passes after construct writes
- `notify setup resolve-events --tool construct --config <config>` resolves the expected dataset `.events.log`
- `infer validate config` and `infer run --dry-run` succeed against the same dataset
- explicit notify dry-run succeeds against the dataset `.events.log`

## Related docs

- Construct workflow and config surface: [../../../construct/docs/README.md](../../../construct/docs/README.md)
- Broader upstream consolidation path: [multi-source-shared-dataset-assembly.md](multi-source-shared-dataset-assembly.md)
- Construct output contract: [../../../construct/docs/reference/outputs.md](../../../construct/docs/reference/outputs.md)
- USR schema contract: [../reference/schema-contract.md](../reference/schema-contract.md)
- USR maintenance merge patterns: [../reference/maintenance.md](../reference/maintenance.md)
- USR workflow map: [workflow-map.md](workflow-map.md)
- Broader feature-matrix branch: [promoter-characterization-feature-matrix.md](promoter-characterization-feature-matrix.md)
- Infer write-back contract: [../../../infer/docs/reference/command-contracts.md](../../../infer/docs/reference/command-contracts.md)
- Notify USR events contract: [../../../../../docs/notify/usr-events.md](../../../../../docs/notify/usr-events.md)
- Docs index: [../../../../../docs/README.md](../../../../../docs/README.md)
- Ops orchestration contracts: [../../../../../docs/operations/orchestration-runbooks.md](../../../../../docs/operations/orchestration-runbooks.md)
