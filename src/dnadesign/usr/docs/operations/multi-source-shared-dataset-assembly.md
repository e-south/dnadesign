## Multi-Source Shared Dataset Assembly

**Type:** runbook
**Plane:** data-plane
**Owner-boundary:** usr
**Entry artifact:** multiple USR-backed input datasets plus explicit merge, carry, and construct intent
**Exit artifact:** one construct-backed downstream USR dataset ready for infer and notify handoff
**Registry-id:** usr.data-plane.multi-source-source-of-truth
**Summary:** Merge multiple USR-backed sources, preserve explicit carry, and hand one construct-backed shared dataset to Infer and Notify.
**Execution-kind:** staged
**Status-kind:** usr-dataset-state

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-19

Use this runbook when inputs already span multiple USR-backed datasets, but downstream consumers should still see one consolidated construct-backed dataset plus one event stream.

Use this runbook to:

- explicit USR merge/carry
- construct realization into one downstream dataset
- handoff into the shared downstream infer/notify continuation once that construct-backed dataset exists

This runbook uses a packaged construct workspace as a local tracer bullet. The
commands below intentionally keep early mutations inside that workspace-local
USR root. For live promoter-study status and continuation, keep the study
record's declared shared USR root authoritative and move or repoint the flow
deliberately before treating the dataset as the cross-tool source of truth.

### Boundary decisions

- USR owns cross-tool consolidation. Upstream producers only need to write valid USR datasets plus any overlays they want preserved.
- `usr maintenance merge` is the only supported consolidation step here; construct does not hide multi-source fan-in inside one job.
- Overlay carry is explicit and narrow: only compact, `id`-keyed, non-reserved namespaces carried with `--carry-namespace`.
- `construct` still owns realization and `construct__*` lineage on the downstream dataset.
- `infer` adds `infer__*` namespaces after construct materializes the downstream dataset.
- `notify` watches the downstream dataset `.events.log`; it does not care which producer created the upstream rows.

### When to use this path

- upstream records already exist in multiple USR datasets
- one upstream source may come from DenseGen while another comes from seeded controls, manual imports, or other USR-backed producers
- one downstream construct dataset should remain the durable shared handoff boundary for infer and downstream consumers
- source labels or other carried overlays must survive the pre-construct consolidation step

### Design stance

- choose one explicit destination dataset for the merge step; `usr maintenance merge` mutates `--dest`
- if mutating one upstream dataset is not acceptable, initialize a fresh destination dataset first and import one source into it before merging additional sources
- start with `--if-duplicate error` and `output.on_conflict=error`
- widen to more templates or more upstream sources only after the tracer bullet passes

## Ordered procedure

### 1) Choose the shared roots and dataset ids

```bash
# Create one disposable workspace root for the tracer-bullet flow.
export WORK_ROOT="$(mktemp -d /tmp/dnadesign-multisource-XXXXXX)" # Allocate a disposable root for the shared workspace and USR datasets.
uv run construct workspace init --id shared_dataset_demo --root "$WORK_ROOT" --profile anchor-template-shared-dataset-demo # Scaffold the packaged construct workspace that will read merged USR inputs.
export WORKSPACE_ROOT="$WORK_ROOT/shared_dataset_demo" # Reuse one workspace path across construct, infer, and notify commands.
export USR_ROOT="$WORKSPACE_ROOT/outputs/usr_datasets" # Keep this tracer-bullet flow inside the packaged workspace-local datasets root.

# Seed packaged control/template inputs used by the tracer bullet.
uv run construct seed anchor-template-demo \
  --root "$USR_ROOT" \
  --manifest "$WORKSPACE_ROOT/inputs/seed_manifest.yaml" # Materialize packaged anchor and template inputs under the workspace-local USR root.

# Reuse explicit dataset ids across merge, construct, infer, and notify.
export PRIMARY_INPUT_DATASET="anchor_parts_demo" # Use the seeded anchor controls as the primary dataset for the tracer bullet.
export EXTRA_INPUT_DATASET="<existing_densegen_or_manual_usr_dataset>" # Replace this with a real upstream dataset that already exists under "$USR_ROOT".
export DOWNSTREAM_DATASET="multi_source_construct_truth_demo" # Reuse one downstream dataset id across construct, infer, and notify.
```

This runbook assumes the upstream datasets already exist in `"$USR_ROOT"`. Their rows may have been produced by different tools or created by earlier import/attach flows.

### 1b) Map those ids to real upstream datasets before validation

The packaged seed step above creates demo datasets such as `anchor_parts_demo` and `template_parts_demo`; it does not create the extra upstream dataset for you.

Before step 2, do one of the following explicitly:

- keep `PRIMARY_INPUT_DATASET="anchor_parts_demo"` and replace `EXTRA_INPUT_DATASET` with a real upstream dataset that already exists under `"$USR_ROOT"`; or
- point both ids at real upstream datasets that already exist under `"$USR_ROOT"`; or
- create/import the missing dataset through your own upstream DenseGen/manual USR flow first, then return here.

Tracer-bullet example when the seeded control dataset should act as the primary input:

```bash
export EXTRA_INPUT_DATASET="<existing_densegen_or_manual_usr_dataset>" # Replace with the real upstream dataset that should be folded into the primary input dataset.
```

For the live `stress_ethanol_cipro_growth` study, replace the packaged demo
inputs with the real shared datasets explicitly:

- `PRIMARY_INPUT_DATASET="mg1655_promoters"`
- `EXTRA_INPUT_DATASET="densegen/study_stress_ethanol_cipro"`
- keep `plasmids` as the pDual-backed template dataset when you repoint the Construct configs in step 3

### 2) Validate and consolidate the upstream USR datasets

```bash
# Confirm both upstream datasets satisfy the active USR registry before merge.
uv run usr --root "$USR_ROOT" validate "$PRIMARY_INPUT_DATASET" --strict # Validate the primary construct-input dataset before it is mutated.
uv run usr --root "$USR_ROOT" validate "$EXTRA_INPUT_DATASET" --strict # Validate the additional upstream dataset before it is merged.

# Merge the extra source into the chosen primary dataset and carry the label overlay.
uv run usr --root "$USR_ROOT" maintenance merge \
  --dest "$PRIMARY_INPUT_DATASET" \
  --src "$EXTRA_INPUT_DATASET" \
  --union-columns \
  --if-duplicate error \
  --carry-namespace usr_label # Merge rows into the chosen primary dataset and carry `usr_label` for surviving source rows.

# Verify the merged input dataset exposes the carried source labels.
uv run usr --root "$USR_ROOT" head "$PRIMARY_INPUT_DATASET" -n 5 \
  --columns id,usr_label__primary,usr_label__aliases # Confirm carried source labels remain queryable after the merge.
```

Expected outcome:

- one merged construct-input dataset now lives at `"$USR_ROOT/$PRIMARY_INPUT_DATASET"`
- its base rows satisfy the active registry
- carried label overlays remain queryable through `usr_label__*`
- `.events.log` records a `merge_datasets` event plus carried namespace counts

### 3) Point the construct workspace at the merged input dataset

The packaged `anchor-template-shared-dataset-demo` workspace is the tracer bullet here. Rewrite both packaged project configs so they read from the merged input dataset and write to one downstream dataset id.

```bash
uv run python - <<'PY' # Rewrite both packaged construct configs so they read the merged input dataset and share one downstream dataset id.
from pathlib import Path
import os

import yaml

workspace_root = Path(os.environ["WORKSPACE_ROOT"])
input_dataset = os.environ["PRIMARY_INPUT_DATASET"]
output_dataset = os.environ["DOWNSTREAM_DATASET"]

for name in ("config.slot_a.window.yaml", "config.slot_b.window.yaml"):
    path = workspace_root / name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["job"]["input"]["source"]["dataset"] = input_dataset
    data["job"]["output"]["target"]["dataset"] = output_dataset
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

registry_path = workspace_root / "construct.workspace.yaml"
registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
for project in registry["workspace"]["projects"]:
    project["contract"]["input_dataset"] = input_dataset
    project["contract"]["output_dataset"] = output_dataset
registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
PY
```

### 4) Validate and materialize construct against the merged input dataset

```bash
export CONSTRUCT_CONFIG="$WORKSPACE_ROOT/config.slot_a.window.yaml" # Keep one construct config path explicit for downstream resolver checks.

uv run construct workspace doctor --workspace "$WORKSPACE_ROOT" # Verify the workspace registry and packaged roots before any construct execution.
uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_a_window --runtime # Validate slot_a against the merged input dataset and shared downstream dataset.
uv run construct workspace validate-project --workspace "$WORKSPACE_ROOT" --project slot_b_window --runtime # Validate slot_b against the merged input dataset and shared downstream dataset.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_a_window --dry-run # Preview slot_a mutations without writing downstream rows.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_b_window --dry-run # Preview slot_b mutations without writing downstream rows.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_a_window # Materialize slot_a rows into the shared downstream dataset.
uv run construct workspace run-project --workspace "$WORKSPACE_ROOT" --project slot_b_window # Materialize slot_b rows into the shared downstream dataset.
```

### 5) Verify the downstream shared dataset

```bash
# Confirm the downstream construct dataset satisfies the active USR registry.
uv run usr --root "$USR_ROOT" validate "$DOWNSTREAM_DATASET" --strict # Confirm the downstream construct dataset satisfies the active USR registry.
# Inspect carried source labels alongside construct lineage on the downstream dataset.
uv run usr --root "$USR_ROOT" head "$DOWNSTREAM_DATASET" -n 10 \
  --columns id,usr_label__primary,construct__input_dataset,construct__input_id,construct__template_id,construct__window_semantics # Inspect carried source labels alongside construct lineage.
```

Expected outcome:

- the downstream dataset is valid under the active registry
- `usr_label__*` from the merged upstream dataset remains available
- `construct__input_*` and `construct__template_id` make every derived row auditable
- one downstream `.events.log` exists at `"$USR_ROOT/$DOWNSTREAM_DATASET/.events.log"`

### 6) Continue through the shared downstream construct-backed handoff

Once the merged upstream dataset has been realized into `"$DOWNSTREAM_DATASET"`, switch to the shared downstream continuation in [construct-infer-shared-dataset-runbook.md](construct-infer-shared-dataset-runbook.md):

```bash
# Reuse the same downstream dataset id under the shared continuation contract.
export DATASET_ID="$DOWNSTREAM_DATASET"
# Continue with the shared infer handoff section.
# See: construct-infer-shared-dataset-runbook.md#5-shared-downstream-continuation-prepare-infer-handoff-against-the-construct-dataset
# Then verify the shared events path.
# See: construct-infer-shared-dataset-runbook.md#6-shared-downstream-continuation-verify-downstream-event-consumption
```

### 7) Continue into scheduler orchestration only after the data-plane passes

Once this path is green locally, use:

- [Infer pressure-test runbook](../../../infer/docs/operations/pressure-test-agnostic-models.md)
- [Ops orchestration runbooks](../../../../../docs/operations/orchestration-runbooks.md)

The Ops control plane is downstream from this runbook. It should not replace the explicit USR merge/carry plus construct/infer handoff steps above.

## Verification checklist

- `usr maintenance merge ... --carry-namespace usr_label` succeeds and emits carried namespace counts
- `usr head "$PRIMARY_INPUT_DATASET" --columns id,usr_label__primary` shows labels from all surviving source rows
- `construct workspace validate-project --runtime` succeeds for every project that targets the shared downstream dataset
- `usr validate "$DOWNSTREAM_DATASET" --strict` passes after construct writes
- the shared downstream continuation in `construct-infer-shared-dataset-runbook.md` succeeds for infer config validation, infer dry-run, and notify event resolution against the same downstream dataset

## Related docs

- Docs index: [../../../../../docs/README.md](../../../../../docs/README.md)
- USR workflow map: [workflow-map.md](workflow-map.md)
- Construct-backed shared dataset handoff: [construct-infer-shared-dataset-runbook.md](construct-infer-shared-dataset-runbook.md)
- USR maintenance merge contract: [../reference/maintenance.md](../reference/maintenance.md)
- Construct workflow docs: [../../../construct/docs/README.md](../../../construct/docs/README.md)
- DenseGen workflow docs: [../../../densegen/docs/README.md](../../../densegen/docs/README.md)
- Infer workflow docs: [../../../infer/docs/README.md](../../../infer/docs/README.md)
- Broader feature-matrix branch: [promoter-characterization-feature-matrix.md](promoter-characterization-feature-matrix.md)
- Notify operator contract: [../../../../../docs/notify/usr-events.md](../../../../../docs/notify/usr-events.md)
- Ops orchestration contracts: [../../../../../docs/operations/orchestration-runbooks.md](../../../../../docs/operations/orchestration-runbooks.md)
