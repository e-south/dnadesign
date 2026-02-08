## DenseGen Binding-Sites Baseline Demo

This demo is the minimal DenseGen path with one constrained variant:

- one binding-sites input (`inputs/sites.csv`, mock TFBS lengths 16–20 bp)
- one unconstrained plan (`baseline`)
- one fixed-promoter plan (`baseline_sigma70`, spacer 16-18 bp)
- no regulator-group constraints
- local parquet outputs under `outputs/tables/`

Use this demo to learn `dense run` behavior before adding constraints.

### 1) Stage a workspace

```bash
uv run dense workspace init --id binding_sites_trial --from-workspace demo_binding_sites --copy-inputs --output-mode local
cd src/dnadesign/densegen/workspaces/binding_sites_trial
CONFIG="$PWD/config.yaml"
```

If `binding_sites_trial` already exists, choose a new `--id` or remove the existing workspace directory first.

### 2) Validate and inspect

```bash
uv run dense validate-config --probe-solver -c "$CONFIG"
uv run dense inspect inputs -c "$CONFIG"
uv run dense inspect plan -c "$CONFIG"
```

### 3) Run and inspect

```bash
uv run dense run --fresh --no-plot -c "$CONFIG"
uv run dense inspect run --library --events -c "$CONFIG"
ls -la outputs/tables/dense_arrays.parquet
```

Expected behavior:

- Stage-A ingests `inputs/sites.csv`.
- Stage-B builds one library per plan.
- `baseline` runs with only TF placements.
- `baseline_sigma70` enforces `sigma70_consensus` (`TTGACA ... TATAAT`, spacer 16-18 bp), so placement maps show fixed-element occupancy in addition to TF occupancy.

### 4) Step-up examples: tune constraints in `config.yaml`

Add a regulator-group requirement:

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

Tune the fixed promoter spacer window:

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
uv run dense run --fresh --no-plot -c "$CONFIG"
uv run dense inspect run --library --events -c "$CONFIG"
```

### 5) Reset and rerun

```bash
uv run dense campaign-reset -c "$CONFIG"
uv run dense run --fresh --no-plot -c "$CONFIG"
uv run dense plot -c "$CONFIG"
```

`campaign-reset` removes `outputs/` and preserves `config.yaml` plus `inputs/`.

---

@e-south
