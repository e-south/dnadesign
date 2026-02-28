## Workspace templates

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This concept page explains how packaged DenseGen workspaces are organized and how to choose between demo and study templates. Read it when you need a consistent path from first-run tutorials to real study workspaces.

For the full DenseGen doc map, use the **[DenseGen documentation index](../index.md)**.

### Template types
This section defines the two template categories and their intended usage.

- `demo_*` templates are didactic and designed for short end-to-end walkthroughs.
- `study_*` templates are larger campaign-style baselines for realistic workloads.

### Packaged templates
This section lists the four packaged templates and their primary intent.

| Template id | Class | Intent | Additional requirements |
| --- | --- | --- | --- |
| `demo_tfbs_baseline` | demo | Minimal lifecycle walkthrough with binding-sites input | Solver backend |
| `demo_sampling_baseline` | demo | PWM artifact sampling walkthrough with two plans | Solver backend plus FIMO |
| `study_constitutive_sigma_panel` | study | Sigma70 constitutive promoter panel with fixed-element combinatorics and LacI/AraC background exclusion | Solver backend plus FIMO |
| `study_stress_ethanol_cipro` | study | Larger stress-response campaign with three plans | Solver backend plus FIMO |

### Tutorial mapping
This section maps template classes to the tutorials that teach them.

- Use **[TFBS baseline tutorial](../tutorials/demo_tfbs_baseline.md)** for `demo_tfbs_baseline`.
- Use **[sampling baseline tutorial](../tutorials/demo_sampling_baseline.md)** for `demo_sampling_baseline`.
- Use **[constitutive sigma panel study tutorial](../tutorials/study_constitutive_sigma_panel.md)** for `study_constitutive_sigma_panel`.
- Use **[stress ethanol and ciprofloxacin study tutorial](../tutorials/study_stress_ethanol_cipro.md)** for `study_stress_ethanol_cipro`.
- Use **[DenseGen to USR to Notify tutorial](../tutorials/demo_usr_notify.md)** when validating watcher delivery on any USR-enabled workspace.

### Standard study workflow
This section gives a reusable command flow for `study_*` templates.

```bash
# Choose a packaged study template.
TEMPLATE_ID="study_constitutive_sigma_panel"

# Resolve repo root and pin workspace root so paths are deterministic.
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORKSPACE_ROOT="$REPO_ROOT/src/dnadesign/densegen/workspaces"

# Create an isolated workspace from that template.
uv run dense workspace init --id study_trial --root "$WORKSPACE_ROOT" --from-workspace "$TEMPLATE_ID" --copy-inputs --output-mode both

# Enter the workspace and define config path.
cd "$WORKSPACE_ROOT/study_trial"
CONFIG="$PWD/config.yaml"

# Validate config and solver availability.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Build Stage-A pools when required by the template.
uv run dense stage-a build-pool --fresh -c "$CONFIG"

# Run, inspect, and render analysis artifacts.
uv run dense run --no-plot -c "$CONFIG"
uv run dense inspect run --events --library -c "$CONFIG"
uv run dense plot -c "$CONFIG"
uv run dense notebook generate -c "$CONFIG"
```

### Study data notes
This section describes where to put private or non-tracked study instructions.

Keep private study notes in workspace-local non-tracked files or external private docs, and keep packaged template docs focused on reusable behavior and schema contracts.

For generic scheduler runs after template setup, use **[DenseGen HPC runbook](../howto/hpc.md)**.
