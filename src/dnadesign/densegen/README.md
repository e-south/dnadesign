## DenseGen — Dense Array Generator

DenseGen builds synthetic DNA arrays by placing TF binding sites under explicit constraints.

If you want one sentence:
DenseGen takes Stage-A site pools, samples Stage-B solver libraries, solves layouts, and writes
run artifacts you can audit.

### What DenseGen is for

- generating quota-bounded sequence libraries with explicit composition constraints
- testing promoter/fixed-element constraints alongside TF placements
- producing diagnostics (attempts, manifests, plots, reports)

### What DenseGen is not for

- canonical storage (use USR)
- webhook/alert delivery (use Notify)

### Event boundary (important)

- DenseGen runtime diagnostics: `outputs/meta/events.jsonl`
- Notify input stream: USR `<usr_root>/<dataset>/.events.log`

Notify consumes USR `.events.log` only.

---

### Contents

- [Pick your first run](#pick-your-first-run)
- [Runtime subprocess flow](#runtime-subprocess-flow)
- [Quick start: binding-sites baseline](#quick-start-binding-sites-baseline)
- [Quick start: canonical PWM path](#quick-start-canonical-pwm-path)
- [Docs map](#docs-map)

---

### Pick your first run

- Fastest learning path: [docs/demo/demo_binding_sites.md](docs/demo/demo_binding_sites.md)
- Canonical PWM workflow: [docs/demo/demo_pwm_artifacts.md](docs/demo/demo_pwm_artifacts.md)
- Full stack (DenseGen -> USR -> Notify): [docs/demo/demo_usr_notify.md](docs/demo/demo_usr_notify.md)

### Runtime subprocess flow

DenseGen runtime is intentionally sequential:

1. Stage-A: build or load input pools (`dense stage-a build-pool`, or auto-built by `dense run`)
2. Stage-B: sample plan-scoped libraries from Stage-A pools (`dense stage-b build-libraries`, or auto-built by `dense run`)
3. Solve: generate arrays under plan constraints to quota (`dense run`)
4. Materialize diagnostics: write tables/manifests/events, then optional plots/report (`dense plot`, `dense report`)

If you keep this order in mind, most diagnostics become easy to interpret:

- pool quality issues are Stage-A
- coverage/composition issues are Stage-B sampling + solver outcomes
- storage/event delivery issues are output-target specific (Parquet vs USR)

### Output modes and handoff paths

- `--output-mode local`: canonical run artifact is `outputs/tables/dense_arrays.parquet`.
- `--output-mode usr`: canonical run artifact is USR dataset `outputs/usr_datasets/<dataset>/records.parquet` plus overlays and `.events.log`.
- `--output-mode both`: writes both sinks and enforces sink-alignment checks during the run.

For `local`, copy `outputs/tables/dense_arrays.parquet`.  
For `usr`, resolve the dataset path from the run, then export:

```bash
# Resolve USR dataset path from the run and export a portable parquet handoff.
EVENTS_PATH="$(uv run dense inspect run --usr-events-path -c "$CONFIG")"
DATASET_PATH="$(dirname "$EVENTS_PATH")"
mkdir -p /tmp/densegen_handoff
uv run usr export "$DATASET_PATH" --fmt parquet --out /tmp/densegen_handoff
```

### Quick start: binding-sites baseline

Run from repo root:

```bash
# Install dependencies from lockfile.
uv sync --locked

# Create a workspace from the binding-sites demo template.
uv run dense workspace init --id binding_sites_trial --from-workspace demo_binding_sites --copy-inputs --output-mode local

# Enter the workspace.
cd src/dnadesign/densegen/workspaces/binding_sites_trial

# Validate config + solver.
uv run dense validate-config --probe-solver

# Run generation from a clean state.
uv run dense run --fresh

# Inspect run diagnostics.
uv run dense inspect run --library --events
```

Local-mode handoff output:

```bash
# Canonical local parquet artifact for downstream copy/use.
ls -lh outputs/tables/dense_arrays.parquet
cp outputs/tables/dense_arrays.parquet /path/to/handoff/binding_sites_trial.parquet
```

### Quick start: canonical PWM path

This path needs MEME Suite (`fimo`) in addition to solver availability.

```bash
# Install Python dependencies.
uv sync --locked

# Install pixi environment (includes MEME tooling).
pixi install

# Confirm FIMO is available.
pixi run fimo --version

# Create workspace from the packaged three-TF demo.
uv run dense workspace init --id meme_three_tfs_trial --from-workspace demo_meme_three_tfs --copy-inputs --output-mode usr

# Enter workspace.
cd src/dnadesign/densegen/workspaces/meme_three_tfs_trial

# Validate config + solver.
uv run dense validate-config --probe-solver

# Build Stage-A pools.
uv run dense stage-a build-pool --fresh

# Run generation without clearing Stage-A pools.
uv run dense run --no-plot

# Inspect library + event diagnostics.
uv run dense inspect run --library --events
```

To continue generation later without editing `config.yaml`, extend quotas at runtime and resume:

```bash
# Resume from existing outputs/state.
uv run dense run --resume --extend-quota 8 --no-plot
```

If a run is interrupted after `--extend-quota`, the next plain `uv run dense run --resume` reuses
the last effective quota target until that target is satisfied.

You can still edit plan quotas in `config.yaml`; runtime `--extend-quota` is for iterative sampling sessions.

---

### Docs map

Guides:
- [docs/guide/workspace.md](docs/guide/workspace.md)
- [docs/guide/inputs.md](docs/guide/inputs.md)
- [docs/guide/sampling.md](docs/guide/sampling.md)
- [docs/guide/generation.md](docs/guide/generation.md)
- [docs/guide/outputs-metadata.md](docs/guide/outputs-metadata.md)
- [docs/guide/postprocess.md](docs/guide/postprocess.md)

Reference:
- [docs/reference/cli.md](docs/reference/cli.md)
- [docs/reference/config.md](docs/reference/config.md)
- [docs/reference/outputs.md](docs/reference/outputs.md)
- [docs/reference/motif_artifacts.md](docs/reference/motif_artifacts.md)

Workflows:
- [docs/workflows/cruncher_pwm_pipeline.md](docs/workflows/cruncher_pwm_pipeline.md)
- [docs/workflows/usr_notify_hpc.md](docs/workflows/usr_notify_hpc.md)
- [docs/workflows/bu_scc_end_to_end.md](docs/workflows/bu_scc_end_to_end.md)

---

@e-south
