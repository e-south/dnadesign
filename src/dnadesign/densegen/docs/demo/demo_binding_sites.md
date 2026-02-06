## DenseGen Vanilla Demo (Binding Sites)

This demo is the minimal DenseGen path:

- one binding-sites input (`inputs/sites.csv`)
- one plan (`baseline`)
- no fixed elements
- no regulator-group constraints
- local parquet outputs under `outputs/tables/`

Use this demo to learn `dense run` behavior before adding constraints.

### 1) Stage a workspace

```bash
uv run dense workspace init --id demo_vanilla --template-id demo_binding_sites_vanilla --copy-inputs --output-mode local
cd src/dnadesign/densegen/workspaces/runs/demo_vanilla
CONFIG="$PWD/config.yaml"
```

If `demo_vanilla` already exists, choose a new `--id` or remove the existing run directory first.

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
- Stage-B builds a full library from the input pool.
- Solver runs with default plan behavior only.

### 4) Step-up examples: add constraints in `config.yaml`

Add a regulator-group requirement:

```yaml
generation:
  plan:
    - name: baseline
      quota: 4
      sampling:
        include_inputs: [basic_sites]
      regulator_constraints:
        groups:
          - name: require_two
            members: [TF_A, TF_B, TF_C]
            min_required: 2
```

Add a fixed promoter element:

```yaml
generation:
  plan:
    - name: baseline
      quota: 4
      sampling:
        include_inputs: [basic_sites]
      fixed_elements:
        promoter_constraints:
          - name: sigma70_consensus
            upstream: TTGACA
            downstream: TATAAT
            spacer_length: [16, 20]
            upstream_pos: [0, 60]
      regulator_constraints:
        groups: []
```

Rerun after each change:

```bash
uv run dense run --fresh --no-plot -c "$CONFIG"
uv run dense inspect run --library --events -c "$CONFIG"
```

---

@e-south
