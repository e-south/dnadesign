## Workspace template concepts

This concept page explains how packaged DenseGen workspaces are organized and how to choose between demo and study templates. Read it when you need a consistent path from first-run tutorials to real study workspaces.

### Template classes
This section defines the two template categories and their intended usage.

- `demo_*` templates are didactic and optimized for short end-to-end walkthroughs.
- `study_*` templates are larger campaign-style baselines for realistic workloads.

### Packaged templates
This section lists the four canonical packaged templates and their primary intent.

| Template id | Class | Intent | Additional requirements |
| --- | --- | --- | --- |
| `demo_tfbs_baseline` | demo | Minimal lifecycle walkthrough with binding-sites input | Solver backend |
| `demo_sampling_baseline` | demo | PWM artifact sampling walkthrough with two plans | Solver backend plus FIMO |
| `study_constitutive_sigma_panel` | study | Constitutive promoter panel using plan-template expansion | Solver backend |
| `study_stress_ethanol_cipro` | study | Larger stress-response campaign with three plans | Solver backend plus FIMO |

### Tutorial mapping
This section maps template classes to the tutorials that teach them.

- Use **[TFBS baseline tutorial](../tutorials/demo_tfbs_baseline.md)** first.
- Use **[sampling baseline tutorial](../tutorials/demo_sampling_baseline.md)** next.
- Use **[DenseGen to USR to Notify tutorial](../tutorials/demo_usr_notify.md)** for event-driven operations.
- Use **[constitutive sigma panel study tutorial](../tutorials/study_constitutive_sigma_panel.md)** for plan-template combinatorics.

### Standard study run pattern
This section gives a reusable command flow for `study_*` templates.

```bash
# Choose a packaged study template.
TEMPLATE_ID="study_constitutive_sigma_panel"

# Create an isolated workspace from that template.
uv run dense workspace init --id study_trial --from-workspace "$TEMPLATE_ID" --copy-inputs --output-mode both

# Enter the workspace and define config path.
cd src/dnadesign/densegen/workspaces/study_trial
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

### Sensitive study notes
This section describes where to put private or non-tracked study instructions.

Keep private study notes in workspace-local non-tracked files or external private docs, and keep packaged template docs focused on reusable behavior and schema contracts.
