## demo_anchor_template_local Runbook

**Workspace Path**
- The workspace path printed by `construct workspace init` (default: `<cwd>/<your-workspace-id>/`).

**Purpose**
- Seed curated anchor-part and template datasets into the workspace-local USR root at `outputs/usr_datasets/`.
- Replace one explicit `anchor_part_short_ref` interval inside the circular `template_backbone_dual_slot` template record.
- Emit either a 1 kb centered window or a full realized template context into separate USR datasets.

**Registry first**
- Inspect the project inventory before running anything:
  - `uv run construct workspace show --workspace .`
  - `uv run construct workspace doctor --workspace .`
- `shared_usr_root` is a repo-relative hint for a deliberate shared mirror, not the default runtime root for this packaged workspace.
- `workspace_usr_root` is the workspace-relative default used by the packaged configs and `runbook.sh`.
- The workspace registry is [construct.workspace.yaml](construct.workspace.yaml).

**File roles**
- primary editable contract surfaces:
  - `construct.workspace.yaml`
  - `config.slot_*.yaml`
  - `inputs/seed_manifest.yaml`
- operator helpers:
  - `runbook.md`
  - `runbook.sh`
- run outputs:
  - `outputs/**`

**Runbook command**
- Default tracer bullet:
  - `./runbook.sh --mode dry-run --config config.slot_a.window.yaml`
- Materialize one selected config:
  - `./runbook.sh --mode run --config config.slot_a.window.yaml`
- Validate all packaged configs:
  - `./runbook.sh --mode validate-all`

`runbook.sh` seeds the curated demo inputs into `outputs/usr_datasets` before validation or execution so the packaged workspace stays self-contained by default. Set `CONSTRUCT_RUNBOOK_USR_ROOT=/path/to/shared/usr/root` only when a shared mirror is intentional.
The wrapper also carries a project-root hint for `uv run --project ...`; override it with `CONSTRUCT_RUNBOOK_PROJECT_ROOT=/path/to/dnadesign` if needed.

**Important note**
- The provided full `template_backbone_dual_slot` record contains two exact `anchor_part_short_ref` matches:
  - `slot_a`: `[2300, 2335)`
  - `slot_b`: `[3621, 3656)`
- The earlier scaffold-only interval `[405, 440)` does not apply to the full `template_backbone_dual_slot` record used here.

### Step-by-step commands

Run these commands from the workspace root:

```bash
set -euo pipefail # Fail fast on errors, unset variables, and pipe failures.

# Bootstrap the local demo inputs and write a manifest with record ids.
uv run construct seed anchor-template-demo \
  --root "$PWD/outputs/usr_datasets" \
  --manifest "$PWD/inputs/seed_manifest.yaml"

# Inspect the workspace registry, verify drift, and inspect seeded labels.
uv run construct workspace show --workspace . # Print workspace registry entries and resolved paths.
uv run construct workspace doctor --workspace . # Check workspace registry/config consistency.
# Inspect seeded anchor-part records and label fields.
uv run usr --root "$PWD/outputs/usr_datasets" head anchor_parts_demo -n 10 \
  --columns id,usr_label__primary,usr_label__aliases,sequence # Show seeded anchor-part labels and aliases.
# Inspect seeded template records and label fields.
uv run usr --root "$PWD/outputs/usr_datasets" head template_parts_demo -n 10 \
  --columns id,usr_label__primary,usr_label__aliases,sequence # Show seeded template labels and aliases.

# Validate and dry-run a 1 kb context realization around the slot_a incumbent.
uv run construct validate config --config "$PWD/config.slot_a.window.yaml" --runtime # Validate slot_a window config with runtime paths.
uv run construct run --config "$PWD/config.slot_a.window.yaml" --dry-run # Preview slot_a window outputs without mutating USR state.

# Materialize the slot_a 1 kb outputs into workspace-local USR.
uv run construct run --config "$PWD/config.slot_a.window.yaml" # Materialize slot_a window outputs.
uv run usr --root "$PWD/outputs/usr_datasets" validate anchor_template_slot_a_window_1kb_demo --strict # Confirm the written output dataset satisfies the active USR registry.

# Validate and dry-run the full-template realization for the same slot.
uv run construct validate config --config "$PWD/config.slot_a.full.yaml" --runtime # Validate slot_a full-template config with runtime paths.
uv run construct run --config "$PWD/config.slot_a.full.yaml" --dry-run # Preview the slot_a full-template output without writing records.
```

### Variations

- Target the second incumbent location:
  - swap `slot_a` for `slot_b`
- Write to a shared USR root instead of the workspace-local one:
  - edit the config `root:` fields intentionally
  - re-run `uv run construct workspace show --workspace .`
  - re-run `validate --runtime` before `run`
- Accumulate multiple construct jobs into one existing output dataset:
  - point multiple configs at the same `output.dataset`
  - keep `output.on_conflict=error` for fail-fast collision detection
  - use `output.on_conflict=ignore` only when idempotent reruns are intentional

### SCC and remote sync

Construct does not define its own remote-sync contract. For BU SCC and other remote USR workflows, use the USR operations docs:

- [USR workflow map](../../../usr/docs/operations/workflow-map.md)
- [USR HPC sync flow](../../../usr/docs/operations/hpc-agent-sync-flow.md)
- [USR sync command contract](../../../usr/docs/operations/sync.md)
