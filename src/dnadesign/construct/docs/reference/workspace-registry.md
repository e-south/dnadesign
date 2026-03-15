## construct workspace registry reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-14

### Purpose

`construct.workspace.yaml` is the project inventory for a construct workspace. Each entry binds one auditable config file to one intended flow and dataset contract.

### Shape

```yaml
workspace:
  id: demo_promoter_swap
  profile: promoter-swap-demo
  description: Construct workspace registry for explicit project inventory.
  roots:
    shared_usr_root: src/dnadesign/usr/datasets
    workspace_usr_root: outputs/usr_datasets
  projects:
    - id: slot_a_window
      config: config.slot_a.window.yaml
      flow: replace-anchor-in-template
      input_dataset: mg1655_promoters
      template_id: pDual-10
      template_dataset: plasmids
      template_record_id: c4f17db3c2dbc17c5cb32c5eec785ea4f091e51d
      output_dataset: pdual10_slot_a_window_1kb_demo
      notes: Windowed promoter swap against slot_a in pDual-10.
```

### Contract

- one workspace registry inventories one construct study
- one project entry maps to one construct config
- project ids must be unique
- config paths must be unique
- dataset ids stay semantic and biological at the USR layer
- template or output routing must not be hidden inside the registry; the config remains authoritative for runtime semantics

### Root precedence

- `workspace.roots.shared_usr_root` is a repo-relative hint for an intentional shared mirror
- `workspace.roots.workspace_usr_root` is the workspace-relative default that packaged workspaces are expected to use
- the runtime still follows the `root:` values in each construct config
- if a config omits `root:`, construct resolves it from the config directory and package defaults, not from `construct.workspace.yaml`
- use `workspace show` to inspect registry hints and `validate --runtime` or `workspace validate-project --runtime` to confirm the actual resolved roots before a write

### Doctoring and execution

Use the registry-backed commands instead of memorizing config paths:

```bash
uv run construct workspace doctor --workspace <workspace-dir>
uv run construct workspace validate-project --workspace <workspace-dir> --project <id> --runtime
uv run construct workspace run-project --workspace <workspace-dir> --project <id> --dry-run
```

`workspace doctor` fails when:

- a project config file is missing
- a project config no longer parses
- `input_dataset`, `output_dataset`, `template_id`, `template_dataset`, or `template_record_id` drift from the registry entry

### Design stance

- keep `construct.workspace.yaml` small and auditable
- represent slot or template matrices as multiple project entries
- do not collapse multiple templates into one construct job just to avoid adding registry entries
