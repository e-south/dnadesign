## DenseGen workspaces directory

This file explains what belongs under `src/dnadesign/densegen/workspaces/` and where to find template docs. Read it when organizing workspace folders or checking what should be committed.

### Core docs
Use these pages as the source of truth for workspace behavior and template intent:

- **[Workspace templates](../docs/concepts/workspace-templates.md)**
- **[Workspace model](../docs/concepts/workspace.md)**
- **[DenseGen documentation index](../docs/index.md)**

### Quick rules
This section provides fast checks for what belongs in this directory and how it is used.

- `demo_*` directories are fast baseline workspaces for guided onboarding and CI pressure tests.
- `study_*` directories are larger template workspaces for real run planning.
- New workspaces created by `dense workspace init` default to this same root.
- `dense workspace init` requires exactly one source option: `--from-workspace` or `--from-config`.
- `archived/` stores historical local artifacts that are not part of active work.

### What belongs here
This directory contains:

- Packaged templates tracked in git (`demo_*`, `study_*`).
- Local workspaces created by `dense workspace init`.
- Historical snapshots under `archived/` when intentionally preserved.

### Packaged workspace flows
This section maps each tracked packaged workspace to runtime intent and its primary didactic tutorial.

| Workspace | Primary flow | High-signal schema keys | Tutorial |
| --- | --- | --- | --- |
| `demo_tfbs_baseline` | Binding-site only smoke run | `densegen.inputs[].type=binding_sites`, `densegen.generation.plan`, `densegen.output.targets=[parquet]` | **[TFBS baseline tutorial](../docs/tutorials/demo_tfbs_baseline.md)** |
| `demo_sampling_baseline` | PWM Stage-A sampling + Stage-B subsampling | `densegen.inputs[].sampling`, `densegen.generation.sampling.pool_strategy=subsample`, `densegen.output.targets=[parquet,usr]` | **[Sampling baseline tutorial](../docs/tutorials/demo_sampling_baseline.md)** |
| `study_constitutive_sigma_panel` | Plan-template expansion with promoter matrices | `densegen.motif_sets`, `densegen.generation.plan_templates`, `densegen.generation.sequence_constraints` | **[Constitutive sigma panel study tutorial](../docs/tutorials/study_constitutive_sigma_panel.md)** |
| `study_stress_ethanol_cipro` | Multi-plan stress study with dual sink outputs | `densegen.generation.plan` (three plans), `densegen.inputs[].sampling.filters.fimo_exclude`, `densegen.output.targets=[parquet,usr]` | **[Stress ethanol and ciprofloxacin study tutorial](../docs/tutorials/study_stress_ethanol_cipro.md)** |

### Notify readiness
This section clarifies which packaged workspaces are intended for USR-event watcher operations.

`study_stress_ethanol_cipro` is the packaged campaign workspace for Notify-bound operations because it writes USR events by default (`output.targets=[parquet,usr]`). For watcher setup and delivery verification, pair the stress tutorial with the **[DenseGen to USR to Notify tutorial](../docs/tutorials/demo_usr_notify.md)**.

### Expected workspace shape
Each workspace should contain:

- `config.yaml`
- `inputs/`
- `outputs/` with stage artifacts (`pools/`, `libraries/`, `tables/`, `meta/`, `plots/`, `notebooks/`, optional `logs/`)

For exact artifact contracts and file semantics, use **[DenseGen outputs reference](../docs/reference/outputs.md)**.

### Git hygiene
Keep production run outputs out of version control. Generated workspaces under `workspaces/` are ignored by default unless explicitly whitelisted.

### Inspect all workspaces
Use this command to inspect runs under this root:

```bash
# Inspect all DenseGen workspaces under the default root.
uv run dense inspect run --root src/dnadesign/densegen/workspaces
```
