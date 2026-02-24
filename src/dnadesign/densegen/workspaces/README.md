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
- For templates that write `usr` outputs, stage a local workspace with `dense workspace init --output-mode both` before `dense run`.
- `archived/` stores historical local artifacts that are not part of active work.

### What belongs here
This directory contains:

- Packaged templates tracked in git (`demo_*`, `study_*`).
- Local workspaces created by `dense workspace init`.
- Historical snapshots under `archived/` when intentionally preserved.

### Packaged workspace flows
This section maps each tracked packaged workspace to runtime intent and its primary didactic tutorial.

| Workspace | Primary flow | High-signal schema keys | Tutorial | Runbook |
| --- | --- | --- | --- | --- |
| `demo_tfbs_baseline` | Binding-site only smoke run | `densegen.inputs[].type=binding_sites`, `densegen.generation.plan`, `densegen.output.targets=[parquet]` | **[TFBS baseline tutorial](../docs/tutorials/demo_tfbs_baseline.md)** | **[`workspaces/demo_tfbs_baseline/runbook.md`](demo_tfbs_baseline/runbook.md)** |
| `demo_sampling_baseline` | PWM Stage-A sampling + Stage-B subsampling | `densegen.inputs[].sampling`, `densegen.generation.sampling.pool_strategy=subsample`, `densegen.output.targets=[parquet,usr]` | **[Sampling baseline tutorial](../docs/tutorials/demo_sampling_baseline.md)** | **[`workspaces/demo_sampling_baseline/runbook.md`](demo_sampling_baseline/runbook.md)** |
| `study_constitutive_sigma_panel` | Deterministic fixed-element matrix expansion with LacI/AraC exclusion | `densegen.motif_sets`, `densegen.generation.plan[].fixed_elements.fixed_element_matrix`, `densegen.inputs[].sampling.filters.fimo_exclude`, `densegen.generation.sequence_constraints` | **[Constitutive sigma panel study tutorial](../docs/tutorials/study_constitutive_sigma_panel.md)** | **[`workspaces/study_constitutive_sigma_panel/runbook.md`](study_constitutive_sigma_panel/runbook.md)** |
| `study_stress_ethanol_cipro` | Multi-condition stress study with fixed-element matrix expansion and dual sink outputs | `densegen.generation.plan[].fixed_elements.fixed_element_matrix`, `densegen.inputs[].sampling.filters.fimo_exclude`, `densegen.output.targets=[parquet,usr]` | **[Stress ethanol and ciprofloxacin study tutorial](../docs/tutorials/study_stress_ethanol_cipro.md)** | **[`workspaces/study_stress_ethanol_cipro/runbook.md`](study_stress_ethanol_cipro/runbook.md)** |

### Fixed-element matrix behavior across packaged workspaces
This section provides a compact behavioral map for `fixed_element_matrix` usage and expected design outcomes.

| Workspace | Matrix usage | Expansion math | Intended outcome |
| --- | --- | --- | --- |
| `demo_tfbs_baseline` | No matrix expansion | `N/A` | Fast binding-site smoke validation with minimal moving parts |
| `demo_sampling_baseline` | No matrix expansion | `N/A` | Stage-A/Stage-B operator flow validation with dual output sinks |
| `study_constitutive_sigma_panel` | One matrix plan (`cross_product`) | `6x8 = 48` concrete plans, aggregate target `48` | Balanced constitutive sigma70 panel for direct core-variant comparisons |
| `study_stress_ethanol_cipro` | Three matrix plans (`cross_product`) with curated selectors | `3x(5x1)=15` concrete plans, aggregate target `200` | Stress-condition campaign with fixed `-10` and curated `-35` sweep for robust dynamic-range exploration |

Policy shared by both matrix-enabled studies:
- Expansion is deterministic at config load and runtime consumes only resolved concrete plans.
- Matrix expansion treats `generation.plan[].sequences` as the base-plan target and requires an exact divisible split across expanded variants.
- Invalid selectors, pairing rules, quota math, name collisions, and cap overflow fail config validation.

### Notify readiness
This section clarifies which packaged workspaces are intended for USR-event watcher operations.

`study_stress_ethanol_cipro` is the packaged campaign workspace for Notify-bound operations because it writes USR events by default (`output.targets=[parquet,usr]`). For watcher setup and delivery verification, pair the stress tutorial with the **[DenseGen to USR to Notify tutorial](../docs/tutorials/demo_usr_notify.md)**.

### Expected workspace shape
Each workspace should contain:

- `config.yaml`
- `runbook.md` (single-command fast path plus canonical step-by-step command sequence)
- `inputs/`
- `outputs/` with stage artifacts (`pools/`, `libraries/`, `tables/`, `meta/`, `plots/`, `notebooks/`, optional `logs/`)

Runbook coupling contract:

- `**Run This Single Command**` must execute the same canonical workflow shown in
  `### Step-by-Step Commands`.
- Optional commands (for example Cruncher artifact refresh or workspace reset)
  must be in explicit optional sections so canonical reproducibility stays
  unambiguous.

For exact artifact contracts and file semantics, use **[DenseGen outputs reference](../docs/reference/outputs.md)**.

### Git hygiene
Keep production run outputs out of version control. Generated workspaces under `workspaces/` are ignored by default unless explicitly whitelisted.

### Inspect all workspaces
Use this command to inspect runs under this root:

```bash
# Inspect all DenseGen workspaces under the default root.
uv run dense inspect run --root src/dnadesign/densegen/workspaces
```
