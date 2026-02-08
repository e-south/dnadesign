## DenseGen Demo: Three-TF PWM Workflow

This is the canonical DenseGen demo for Cruncher-derived PWM artifacts (`lexA`, `cpxR`, `baeR`) using a USR output sink.
If you are new to DenseGen, run [demo_binding_sites.md](demo_binding_sites.md) first.

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
10. [Plot and report](#9-plot-and-report)
11. [Reset workspace](#10-reset-workspace)
12. [Troubleshooting](#11-troubleshooting)

## Prerequisites

You need:

1. Python dependencies
2. MEME Suite (`fimo` on `PATH`)
3. A supported solver backend (`CBC` or `GUROBI`)

From repo root:

```bash
uv sync --locked
pixi install
pixi run fimo --version
```

## 1) Stage workspace

```bash
uv run dense workspace init --id demo_pwm --template-id demo_meme_three_tfs --copy-inputs --output-mode usr
cd src/dnadesign/densegen/workspaces/runs/demo_pwm
```

If `demo_pwm` already exists, choose a new `--id` or remove the existing run directory first.

`workspace init --output-mode usr` seeds `outputs/usr_datasets/registry.yaml` in the workspace.

Optional wrapper (if you want shorter commands):

```bash
unalias dense 2>/dev/null
# pixi-backed
dense() { pixi run dense -- "$@"; }
# or uv-backed
# dense() { uv run dense "$@"; }
```

Set a config variable once and pass it explicitly:

```bash
CONFIG="$PWD/config.yaml"
```

## 2) Validate and inspect

```bash
dense validate-config --probe-solver -c "$CONFIG"
dense inspect inputs -c "$CONFIG"
dense inspect plan -c "$CONFIG"
dense inspect config -c "$CONFIG"
```

## 3) Build Stage-A pools

The demo ships motif artifact JSON files in `inputs/motif_artifacts/`.
To regenerate those files from source data, use:

- [../workflows/cruncher_pwm_pipeline.md](../workflows/cruncher_pwm_pipeline.md)

Build Stage-A pools:

```bash
dense stage-a build-pool --fresh -c "$CONFIG"
```

Quick Stage-A plot:

```bash
dense plot --only stage_a_summary -c "$CONFIG"
```

## 4) Run generation to quota

```bash
dense run --fresh --no-plot -c "$CONFIG"
```

Useful debug flags:

```bash
dense run --show-tfbs --no-plot -c "$CONFIG"
dense run --show-solutions --no-plot -c "$CONFIG"
```

## 5) Resume with no changes (safety check)

Rerunning with `--resume` should be a no-op once quota is met:

```bash
dense run --resume --no-plot -c "$CONFIG"
```

## 6) Increase quota and resume

To generate more sequences in the same workspace:

1. Increase quota fields in `config.yaml` (example: `generation.quota` 48 -> 52 and each plan quota 12 -> 13).
2. Resume with explicit quota-growth intent:

```bash
dense run --resume --allow-quota-increase --no-plot -c "$CONFIG"
```

Rules:

- `--allow-quota-increase` only accepts quota-based plans.
- Only `generation.quota` and `generation.plan[*].quota` may change.
- Quotas must not decrease.
- Any other config change fails fast and requires `dense run --fresh` or `dense campaign-reset`.

## 7) Inspect outputs

```bash
dense inspect run --events --library -c "$CONFIG"
```

Key files:

- DenseGen runtime artifacts:
  - `outputs/meta/run_manifest.json`
  - `outputs/meta/events.jsonl`
  - `outputs/pools/pool_manifest.json`
- USR dataset artifacts:
  - `outputs/usr_datasets/demo_pwm/records.parquet`
  - `outputs/usr_datasets/demo_pwm/.events.log`
  - `outputs/usr_datasets/demo_pwm/_derived/densegen/part-*.parquet`

## 8) Wire Notify to a real endpoint (deployed pressure test)

Notify setup and secret-handling details live in:

- `../../../../../docs/notify/usr_events.md`

Slack-first pressure test flow:

```bash
read -rsp "Webhook URL: " DENSEGEN_WEBHOOK; echo
export DENSEGEN_WEBHOOK
```

```bash
EVENTS_PATH="$(dense inspect run --usr-events-path -c "$CONFIG")"

uv run notify profile wizard \
  --profile outputs/notify.profile.json \
  --provider slack \
  --events "$EVENTS_PATH" \
  --secret-source auto \
  --preset densegen
```

```bash
uv run notify profile doctor --profile outputs/notify.profile.json
uv run notify usr-events watch --profile outputs/notify.profile.json --dry-run
```

If payloads look correct, remove `--dry-run` to send for real.

Then trigger more events by increasing quota and resuming:

```bash
dense run --resume --allow-quota-increase --no-plot -c "$CONFIG"
```

Email note: use `--provider generic` with your email relay webhook endpoint.

## 9) Plot and report

If your environment has Matplotlib cache restrictions, set `MPLCONFIGDIR` first:

```bash
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"
```

Then run:

```bash
dense plot --only stage_a_summary,placement_map -c "$CONFIG"
dense report --plots include -c "$CONFIG"
```

## 10) Reset workspace

```bash
dense campaign-reset -c "$CONFIG"
```

This removes `outputs/` and keeps `config.yaml` plus `inputs/`.

## 11) Troubleshooting

### `fimo: command not found`

```bash
pixi run fimo --version
```

### Solver backend not available

```bash
dense validate-config --probe-solver -c "$CONFIG"
```

### USR registry missing

DenseGen fails fast if `outputs/usr_datasets/registry.yaml` is missing or incompatible.
`workspace init --output-mode usr` seeds this file. If you changed `output.usr.root`, ensure the new root has a valid `registry.yaml`.

### Resume/config mismatch

If config changed beyond allowed quota growth, start clean:

```bash
dense run --fresh --no-plot -c "$CONFIG"
# or
dense campaign-reset -c "$CONFIG"
```

### Regulator label mismatch

Check exact labels used by Stage-A:

```bash
dense inspect inputs --show-motif-ids -c "$CONFIG"
```

---

@e-south
