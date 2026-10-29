<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[Construct docs](../README.md)

# Construct workspace registry reference

## Purpose

`construct.workspace.yaml` is the project inventory for a construct workspace. Each entry names a configuration file and the input, template and output identities expected from it.

## Shape

```yaml
workspace:
  id: demo_anchor_template
  profile: anchor-template-demo
  description: Construct workspace registry for explicit project inventory.
  roots:
    shared_usr_root: src/dnadesign/seqledger/datasets
    workspace_usr_root: outputs/usr_datasets
  projects:
    - id: slot_a_window
      artifacts:
        config:
          path: config.slot_a.window.yaml
          job_id: anchor_template_slot_a_window_1kb
      contract:
        input_dataset: anchor_parts_demo
        template:
          id: template_backbone_dual_slot
          dataset: template_parts_demo
          record_id: c4f17db3c2dbc17c5cb32c5eec785ea4f091e51d
        output_dataset: anchor_template_slot_a_window_1kb_demo
      notes: Windowed anchor placement against slot_a in the packaged dual-slot template.
```

## Contract

- one workspace registry inventories one construct study
- one project entry maps to one construct config
- project ids must be unique
- `project.artifacts.config.path` values must be unique
- every project must track `project.artifacts.config.job_id`; registry/config job-id drift is rejected
- every project must declare a `contract:` block; flat legacy fields are rejected
- when `project.contract.template` is present, it must include `id`, `dataset`, and `record_id` together
- dataset ids stay semantic and biological at the SeqLedger layer
- `project.contract` records expected identities for validation; the job config determines what runs

## Root precedence

- `workspace.roots.shared_usr_root` is a repo-relative hint for an intentional shared mirror
- `workspace.roots.workspace_usr_root` is the workspace-relative default that packaged workspaces are expected to use
- the runtime still follows the `root:` values in each construct config
- SeqLedger-backed jobs require an explicit input root; template and output roots may inherit it
- relative roots resolve from the configuration directory, not from the registry
- use `workspace show` to inspect registry hints and `validate --runtime` or `workspace validate-project --runtime` to confirm the actual resolved roots before a write

## Check and run a project

Use the registry-backed commands instead of memorizing config paths:

```bash
# Replace the workspace path and project ID with your own values.
construct workspace doctor --workspace <workspace-dir>
# Resolve actual inputs and output paths before execution.
construct workspace validate-project \
  --workspace <workspace-dir> --project <id> --runtime
# Preview the output write.
construct workspace run-project \
  --workspace <workspace-dir> --project <id> --dry-run
```

`workspace doctor` fails when:

- a project config file is missing
- a project config no longer parses
- `project.artifacts.config.job_id` drifts from `job.id`
- `project.contract.input_dataset`, `project.contract.output_dataset`, or `project.contract.template.*` drift from the config

## Organize related jobs

Use one project entry per configuration. Several entries can share an output
dataset when their conflict policy permits it. Keep placement and window
settings in the job configuration; use the registry to find projects and check
their dataset identities before execution.
