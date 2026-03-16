## demo_promoter_swap_pdual10_source_of_truth Runbook

**Workspace Path**
- `src/dnadesign/construct/workspaces/<your-workspace-id>/`

**Purpose**
- Seed curated promoter and plasmid datasets into the workspace-local USR root at `outputs/usr_datasets/`.
- Replace the explicit `J23105` incumbents in both `slot_a` and `slot_b` with canonical promoter inputs.
- Materialize both 1 kb window realizations into one semantic dataset, `pdual10_source_of_truth_demo`.
- Keep upstream `usr_label__primary` / `usr_label__aliases` visible on derived rows so analyst-facing reads stay human-scannable.

**Registry first**
- Inspect the project inventory before running anything:
  - `uv run construct workspace show --workspace .`
  - `uv run construct workspace doctor --workspace .`
- `shared_usr_root` is a repo-relative hint for an intentional shared mirror, not the default runtime root for this packaged workspace.
- `workspace_usr_root` is the workspace-relative default used by the packaged configs and `runbook.sh`.
- The workspace registry is [construct.workspace.yaml](construct.workspace.yaml).

**File roles**
- primary editable contract surfaces:
  - `construct.workspace.yaml`
  - `config.slot_*.window.yaml`
  - `inputs/seed_manifest.yaml`
- operator helpers:
  - `runbook.md`
  - `runbook.sh`
- run outputs:
  - `outputs/**`

**Runbook command**
- Seed only:
  - `./runbook.sh --mode seed`
- Validate both packaged projects:
  - `./runbook.sh --mode validate-all`
- Dry-run both packaged projects:
  - `./runbook.sh --mode dry-run-all`
- Materialize the shared source-of-truth dataset:
  - `./runbook.sh --mode run-all`

`runbook.sh` seeds the curated demo inputs into `outputs/usr_datasets` before validation or execution so the packaged workspace stays self-contained by default. Set `CONSTRUCT_RUNBOOK_USR_ROOT=/path/to/shared/usr/root` only when a shared mirror is intentional.
The wrapper also carries a project-root hint for `uv run --project ...`; override it with `CONSTRUCT_RUNBOOK_PROJECT_ROOT=/path/to/dnadesign` if needed.

**Important note**
- The provided full `pDual-10` record contains two exact `J23105` matches:
  - `slot_a`: `[2300, 2335)`
  - `slot_b`: `[3621, 3656)`
- This packaged source-of-truth demo only uses the two 1 kb window projects because they are the intended infer/notify handoff shape.

### Step-by-step commands

Run these commands from the workspace root:

```bash
set -euo pipefail # Fail fast on errors, unset variables, and pipe failures.

# Bootstrap the local demo inputs and write a manifest with record ids.
uv run construct seed promoter-swap-demo \
  --root "$PWD/outputs/usr_datasets" \
  --manifest "$PWD/inputs/seed_manifest.yaml"

# Inspect the workspace registry, verify drift, and inspect seeded labels.
uv run construct workspace show --workspace . # Print workspace registry entries and resolved paths.

# Re-check the packaged registry/config contract before running anything.
uv run construct workspace doctor --workspace .
# Show seeded promoter labels and aliases.
uv run usr --root "$PWD/outputs/usr_datasets" head mg1655_promoters -n 10 --columns id,usr_label__primary,usr_label__aliases,sequence

# Validate both packaged projects before any write.
uv run construct workspace validate-project --workspace . --project slot_a_window --runtime # Validate slot_a runtime roots and template resolution.
uv run construct workspace validate-project --workspace . --project slot_b_window --runtime # Validate slot_b runtime roots and template resolution.

# Dry-run both projects into the shared output contract.
uv run construct workspace run-project --workspace . --project slot_a_window --dry-run # Preview slot_a writes without mutating USR state.
uv run construct workspace run-project --workspace . --project slot_b_window --dry-run # Preview slot_b writes without mutating USR state.

# Materialize both windows into one semantic dataset.
uv run construct workspace run-project --workspace . --project slot_a_window # Materialize slot_a rows into the shared dataset.
uv run construct workspace run-project --workspace . --project slot_b_window # Materialize slot_b rows into the shared dataset.

# Verify the shared dataset and inspect carried-through labels plus construct lineage.
uv run usr --root "$PWD/outputs/usr_datasets" validate pdual10_source_of_truth_demo --strict # Confirm the shared dataset satisfies the active USR registry.
# Inspect the shared dataset with upstream labels plus construct lineage context.
uv run usr --root "$PWD/outputs/usr_datasets" head pdual10_source_of_truth_demo -n 10 --columns id,usr_label__primary,construct__input_dataset,construct__input_id,construct__template_id,construct__window_semantics
```

### Variations

- Write to a shared USR root instead of the workspace-local one:
  - edit the config `root:` fields intentionally
  - re-run `uv run construct workspace show --workspace .`
  - re-run `validate-project --runtime` for both projects before `run-project`
- Add more construct projects into the same semantic dataset:
  - keep each project as its own `construct.workspace.yaml` entry
  - keep `output.on_conflict=error` for fail-fast collision detection
  - only widen the shared dataset when the consolidated downstream contract is intentional

### Cross-tool continuation

For the infer and downstream handoff that continues from this workspace, use the shared runbooks:

- [Construct -> USR -> Infer source-of-truth demo](../../../usr/docs/operations/construct-infer-source-of-truth-demo.md)
- [Promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md)
