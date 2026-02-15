## DenseGen Binding-Sites Baseline Demo

This is the simplest DenseGen walkthrough. It is the right starting point if you want to
learn the runtime flow before you add PWM mining or heavy constraints.

This demo uses:

- one binding-sites input (`inputs/sites.csv`, mock TFBS lengths 16-20 bp)
- one unconstrained plan (`baseline`)
- one fixed-promoter plan (`baseline_sigma70`, spacer 16-18 bp)
- no regulator-group constraints
- local Parquet output under `outputs/tables/`

By the end, you will have run a full generation, inspected diagnostics, and seen how
constraint changes alter behavior.

Subprocess map for this demo:

1. Stage workspace + validate config
2. Run (`dense run --fresh`) which executes Stage-A ingest -> Stage-B sampling -> solve loop
3. Inspect run diagnostics
4. Iterate config changes and rerun

### Contents
1. [Stage a workspace](#1-stage-a-workspace)
2. [Validate and inspect](#2-validate-and-inspect)
3. [Run and inspect](#3-run-and-inspect)
4. [Step-up examples: tune constraints in `config.yaml`](#4-step-up-examples-tune-constraints-in-configyaml)
5. [Reset and rerun](#5-reset-and-rerun)

### 1) Stage a workspace

```bash
# Create a fresh workspace from the packaged demo template.
uv run dense workspace init --id binding_sites_trial --from-workspace demo_binding_sites --copy-inputs --output-mode local

# Move into the new workspace.
cd src/dnadesign/densegen/workspaces/binding_sites_trial

# Reuse this path in the next commands.
CONFIG="$PWD/config.yaml"
```

If `binding_sites_trial` already exists, choose a new `--id` or remove the existing workspace directory first.

### 2) Validate and inspect

```bash
# Validate schema and verify solver availability.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved Stage-A inputs.
uv run dense inspect inputs -c "$CONFIG"

# Inspect resolved generation plan/quota settings.
uv run dense inspect plan -c "$CONFIG"
```

### 3) Run and inspect

```bash
# Run generation from a clean outputs directory (no plots yet).
uv run dense run --fresh --no-plot -c "$CONFIG"

# Inspect run diagnostics with library + event summaries.
uv run dense inspect run --library --events -c "$CONFIG"

# Confirm the main output table exists.
ls -la outputs/tables/records.parquet
```

Expected behavior:

- Stage-A ingests `inputs/sites.csv`.
- Stage-B builds one library per plan.
- `baseline` runs with only TF placements.
- `baseline_sigma70` enforces `sigma70_consensus` (`TTGACA ... TATAAT`, spacer 16-18 bp), so placement maps show fixed-element occupancy in addition to TF occupancy.

### 4) Step-up examples: tune constraints in `config.yaml`

Now edit `config.yaml` and try one change at a time.

Example A: add a regulator-group requirement:

```yaml
generation:
  plan:
    - name: baseline
      quota: 3
      sampling:
        include_inputs: [basic_sites]
      regulator_constraints:
        groups:
          - name: require_two
            members: [TF_A, TF_B, TF_C]
            min_required: 2
```

Example B: tune the fixed promoter spacer window:

```yaml
generation:
  plan:
    - name: baseline_sigma70
      quota: 2
      sampling:
        include_inputs: [basic_sites]
      fixed_elements:
        promoter_constraints:
          - name: sigma70_consensus
            upstream: TTGACA
            downstream: TATAAT
            spacer_length: [16, 18]
      regulator_constraints:
        groups: []
```

Rerun after each change:

```bash
# Re-run from a clean state after editing config.yaml.
uv run dense run --fresh --no-plot -c "$CONFIG"

# Compare diagnostics to the previous run.
uv run dense inspect run --library --events -c "$CONFIG"
```

### 5) Reset and rerun

```bash
# Remove outputs only (keep config + inputs).
uv run dense campaign-reset -c "$CONFIG"

# Run generation again.
uv run dense run --fresh --no-plot -c "$CONFIG"

# Render plots for this run.
uv run dense plot -c "$CONFIG"
```

`campaign-reset` removes `outputs/` and preserves `config.yaml` plus `inputs/`.

Next step: move to the PWM demo at
[demo_pwm_artifacts.md](demo_pwm_artifacts.md).

---

@e-south
