## DenseGen Demo: Sampling Baseline (PWM Artifacts)

This is the canonical DenseGen sampling baseline for PWM-driven generation.
It uses three TFs (`lexA`, `cpxR`, `baeR`) and writes results to local USR + parquet outputs.
The packaged baseline includes two plans (`ethanol`, `ciprofloxacin`) for quick end-to-end cycles.

If you are new to DenseGen, run [demo_tfbs_baseline.md](demo_tfbs_baseline.md) first.

### What this demo teaches

- how to stage a workspace from a packaged template
- how to build Stage-A pools from motif artifacts
- how to run generation to quota, then resume safely
- how to inspect outputs, wire Notify, and render notebook summaries

Subprocess map for this demo:

1. Stage-A pool build from motif artifacts
2. Stage-B library sampling per plan
3. Solver loop to plan quotas
4. Output materialization to Parquet/USR + plots/notebook

### Contents

1. [Prerequisites](#prerequisites)
2. [Stage workspace](#1-stage-workspace)
3. [Validate and inspect](#2-validate-and-inspect)
4. [Build Stage-A pools](#3-build-stage-a-pools)
5. [Run generation to quota](#4-run-generation-to-quota)
6. [Resume with no changes (safety check)](#5-resume-with-no-changes-safety-check)
7. [Increase quota and resume](#6-increase-quota-and-resume)
8. [Inspect outputs](#7-inspect-outputs)
9. [Wire Notify to a real endpoint](#8-wire-notify-to-a-real-endpoint-deployed-pressure-test)
10. [Plot and notebook](#9-plot-and-notebook)
11. [Reset workspace](#10-reset-workspace)
12. [Troubleshooting](#11-troubleshooting)

## Prerequisites

You need:

1. Python dependencies
2. MEME Suite (`fimo` on `PATH`)
3. A supported solver backend (`CBC` or `GUROBI`)

Run from repo root:

```bash
# Install Python dependencies from the lockfile.
uv sync --locked

# Install pixi environment (includes MEME Suite tooling).
pixi install

# Confirm FIMO is available.
pixi run fimo --version
```

## 1) Stage workspace

```bash
# Create a new workspace from the packaged sampling-baseline demo.
uv run dense workspace init --id sampling_baseline_trial --from-workspace demo_sampling_baseline --copy-inputs --output-mode both

# Move into the workspace.
cd src/dnadesign/densegen/workspaces/sampling_baseline_trial

# Store config path once for the rest of the demo.
CONFIG="$PWD/config.yaml"
```

If `sampling_baseline_trial` already exists, choose a new `--id` or remove that workspace first.

`workspace init --output-mode both` also seeds `outputs/usr_datasets/registry.yaml`
when a repo registry seed file is available.

If `fimo` is only available through pixi on your machine, run the same command flow with
`pixi run dense ...` instead of `uv run dense ...`.

## 2) Validate and inspect

```bash
# Validate config structure and probe solver availability.
uv run dense validate-config --probe-solver -c "$CONFIG"

# Inspect resolved Stage-A inputs.
uv run dense inspect inputs -c "$CONFIG"

# Inspect resolved plan/quota settings.
uv run dense inspect plan -c "$CONFIG"

# Inspect full resolved runtime config summary.
uv run dense inspect config -c "$CONFIG"
```

Packaged `demo_sampling_baseline` defaults to `solver.backend: CBC`.
If you want GUROBI on your machine, update `config.yaml` before running:

```yaml
solver:
  backend: GUROBI
  strategy: iterate
  time_limit_seconds: 5
  # threads: 4
```

## 3) Build Stage-A pools

With `--copy-inputs`, this workspace stores motif artifact JSON files under `inputs/`.
The config paths are rewritten to those copied files.

If you need to regenerate those files from upstream data, follow:
- [../workflows/cruncher_pwm_pipeline.md](../workflows/cruncher_pwm_pipeline.md)

Now build Stage-A pools:

```bash
# Build pools from motif artifacts, replacing any prior pool outputs.
uv run dense stage-a build-pool --fresh -c "$CONFIG"
```

What this packaged demo does in Stage-A:

- For each PWM input (`lexA`, `cpxR`, `baeR`), DenseGen mines **150,000** candidates and
  retains **100** TFBS rows after scoring, dedupe, and MMR selection.
- For the `background` input, DenseGen mines **600,000** candidates and retains **200** rows.
- In plain terms:
  - `mining.budget.candidates` = search effort
  - `n_sites` = final retained pool size

This split lets you increase search coverage without forcing the solver to handle a much
larger retained pool.

Optional quick check plot:

```bash
# Render only Stage-A summary plots.
uv run dense plot --only stage_a_summary -c "$CONFIG"
```

## 4) Run generation to quota

```bash
# Run using the Stage-A pools you just built and skip auto-plotting for speed.
uv run dense run --no-plot -c "$CONFIG"
```

`dense run --fresh` clears prior outputs (including Stage-A pool artifacts). Use `--fresh`
only when you intentionally want a full rebuild.

`demo_sampling_baseline` uses `logging.progress_style: auto`, so progress output adapts to
terminal capabilities automatically.

Useful debug variants:

```bash
# Show TFBS names in progress output.
uv run dense run --show-tfbs --no-plot -c "$CONFIG"

# Show full solution sequences in progress output.
uv run dense run --show-solutions --no-plot -c "$CONFIG"
```

## 5) Resume with no changes (safety check)

After quota is met, a resume run should be a no-op:

```bash
# Confirm resume behavior with no config changes.
uv run dense run --resume --no-plot -c "$CONFIG"
```

## 6) Increase quota and resume

To generate more sequences in the same workspace:

1. edit `config.yaml`
2. increase one or more `generation.plan[*].quota` values (for example 6 -> 9)
3. resume

```bash
# Continue from prior state after quota-only plan increase.
uv run dense run --resume --no-plot -c "$CONFIG"
```

Rules:

- plan quotas must not decrease
- quota-only plan increases are allowed on resume
- any other config change requires a fresh run (`uv run dense run --fresh`) or reset (`uv run dense campaign-reset`)

## 7) Inspect outputs

```bash
# Show run summary with event and library diagnostics.
uv run dense inspect run --events --library -c "$CONFIG"
```

Key files:

- DenseGen runtime artifacts:
  - `outputs/meta/run_manifest.json`
  - `outputs/meta/events.jsonl`
  - `outputs/pools/pool_manifest.json`
- Parquet records surface (used by notebook generation):
  - `outputs/tables/records.parquet`
- USR dataset artifacts:
  - `outputs/usr_datasets/densegen/demo_sampling_baseline/records.parquet`
  - `outputs/usr_datasets/densegen/demo_sampling_baseline/.events.log`
  - `outputs/usr_datasets/densegen/demo_sampling_baseline/_derived/densegen/part-*.parquet`

## 8) Wire Notify to a real endpoint (deployed pressure test)

Notify setup details (profiles, secrets, spool/drain) are in:
- [../../../../../docs/notify/usr-events.md](../../../../../docs/notify/usr-events.md)
- [Command anatomy: `notify setup slack`](../../../../../docs/notify/usr-events.md#command-anatomy-notify-setup-slack)

Slack-first flow:

```bash
# Capture webhook URL securely in the current shell.
read -rsp "Webhook URL: " DENSEGEN_WEBHOOK; echo

# Export so Notify can read it by env var.
export DENSEGEN_WEBHOOK
```

```bash
# Keep Notify artifacts in this run workspace.
NOTIFY_DIR="outputs/notify"

# Create a Notify profile from workspace config (auto-resolves USR events path).
uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --profile "$NOTIFY_DIR/profile.json" \
  --cursor "$NOTIFY_DIR/cursor" \
  --spool-dir "$NOTIFY_DIR/spool" \
  --secret-source env \
  --url-env DENSEGEN_WEBHOOK \
  --policy densegen
```

```bash
# Validate profile wiring.
uv run notify profile doctor --profile "$NOTIFY_DIR/profile.json"

# Preview payloads without sending.
uv run notify usr-events watch --profile "$NOTIFY_DIR/profile.json" --dry-run
```

If payloads look correct, remove `--dry-run` to send for real.

To generate additional events, increase quota and resume:

```bash
# Emit more USR events by continuing generation.
uv run dense run --resume --no-plot -c "$CONFIG"
```

Email note: use `--provider generic` with your email relay webhook endpoint.

## 9) Plot and notebook

DenseGen uses a repo-local Matplotlib cache at `.cache/matplotlib/densegen`.
Set `MPLCONFIGDIR` only if you need to override this path.

```bash
# Render core plots for this run.
uv run dense plot --only stage_a_summary,placement_map -c "$CONFIG"

# Generate workspace notebook and launch it.
uv run dense notebook generate -c "$CONFIG"
uv run dense notebook run -c "$CONFIG"

# Remote/non-GUI shell: launch without browser auto-open.
uv run dense notebook run --headless -c "$CONFIG"

# Optional: launch editable marimo cells instead of app mode.
uv run dense notebook run --mode edit -c "$CONFIG"
```

## 10) Reset workspace

```bash
# Delete outputs and run state only.
uv run dense campaign-reset -c "$CONFIG"
```

This keeps `config.yaml` and `inputs/` in place.

## 11) Troubleshooting

### `fimo: command not found`

```bash
# Verify FIMO from pixi environment.
pixi run fimo --version
```

### Solver backend not available

```bash
# Probe solver and fail fast if unavailable.
uv run dense validate-config --probe-solver -c "$CONFIG"
```

### `logging.progress_style=screen requires ...`

`progress_style: screen` requires an interactive terminal with cursor controls.

For interactive shells:

```bash
# Set terminal type and rerun.
export TERM=xterm-256color
uv run dense run --fresh --no-plot -c "$CONFIG"
```

For non-interactive runs (CI, redirected logs), set `densegen.logging.progress_style: stream`.

### USR registry missing

DenseGen fails fast if `outputs/usr_datasets/registry.yaml` is missing or incompatible.
`workspace init --output-mode usr|both` seeds this file. If you changed `output.usr.root`,
ensure that new root has a valid `registry.yaml`.

### Notebook records source resolution

`dense notebook generate` reads one parquet records source selected by output wiring:

- one sink in `output.targets` -> that sink (`parquet` or `usr`)
- two sinks in `output.targets` -> `plots.source`

This demo uses `--output-mode both` with `plots.source: usr`, so notebook previews read
`outputs/usr_datasets/<dataset>/records.parquet`.

### Resume/config mismatch

If config changed beyond allowed quota growth, restart clean:

```bash
# Fresh run path.
uv run dense run --fresh --no-plot -c "$CONFIG"

# Full outputs reset path.
uv run dense campaign-reset -c "$CONFIG"
```

### Regulator label mismatch

```bash
# Inspect exact Stage-A labels used for constraint matching.
uv run dense inspect inputs --show-motif-ids -c "$CONFIG"
```

---

@e-south
