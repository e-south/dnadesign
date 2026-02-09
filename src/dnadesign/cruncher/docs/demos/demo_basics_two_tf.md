# Two-TF demo (end-to-end)

## Contents
- [Overview](#overview)
- [TL;DR (copy/paste e2e)](#tldr-copypaste-e2e)
- [Demo setup](#demo-setup)
- [Reset demo](#reset-demo)
- [Cache motifs and sites](#cache-motifs-and-sites)
- [Lock + parse](#lock--parse)
- [Sample + analyze](#sample--analyze)
- [Fast rerun after config edits](#fast-rerun-after-config-edits)
- [Inspect results](#inspect-results)
- [Optional: motif discovery](#optional-motif-discovery)
- [Related docs](#related-docs)

## Overview

This demo designs fixed-length sequences that satisfy two PWMs (LexA + CpxR). It follows the full lifecycle:

1. cache inputs
2. lock
3. parse
4. sample
5. analyze

Cruncher scores each TF by the best PWM match anywhere in the sequence on either strand (when `objective.bidirectional=true`). It optimizes the weakest TF by default (`objective.combine=min`) and selects diverse elites via TFBS-core MMR.
The bundled demo config uses targeted insertion proposals plus adaptive move weights/proposal sizing with strict PT adaptation (`n_temps=4`, `temp_max=14`) for stable optimization diagnostics.

For the full intent, lifecycle, and config mapping, see [Intent + lifecycle](../guides/intent_and_lifecycle.md).

## TL;DR (copy/paste e2e)

Use this block for a clean, repeatable end-to-end run that regenerates plots.

```bash
set -euo pipefail
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
CONFIG="$PWD/config.yaml"

# Clean reset (robust to stale parse/lock/run artifacts)
rm -rf outputs
rm -rf .cruncher/parse .cruncher/locks .cruncher/campaigns
rm -f .cruncher/run_index.json

pixi run cruncher -- fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
pixi run cruncher -- fetch sites  --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
pixi run cruncher -- lock -c "$CONFIG"
pixi run cruncher -- parse --force-overwrite -c "$CONFIG"
pixi run cruncher -- sample --force-overwrite -c "$CONFIG"
pixi run cruncher -- analyze --summary -c "$CONFIG"

find outputs -type f -name 'plot__*.png' | sort
```

If `sample` fails due elite filters, adjust `sample.elites.filter.min_per_tf_norm`,
`sample.sequence_length`, or `sample.budget.draws`, then rerun.

## Demo setup

- Workspace: `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/`
- Config: `config.yaml`
- Output root: `outputs/`

```bash
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
CONFIG="$PWD/config.yaml"

# Convenience wrapper (pixi is the default in this repo)
cruncher() { pixi run cruncher -- "$@"; }
```

## Reset demo

Run this before re-running the full demo from scratch.
It clears run artifacts and workspace state while leaving shared catalog caches intact.

```bash
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
rm -rf outputs
rm -rf .cruncher/parse .cruncher/locks .cruncher/campaigns
rm -f .cruncher/run_index.json
```

## Cache motifs and sites

The demo ships with local MEME motifs and MEME BLOCKS sites.

```bash
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites  --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
```

With `catalog.pwm_source: matrix` in this demo, `lock`/`parse` select matrix motifs from the lockfile.
Fetched sites are cached for site-based workflows and diagnostics, but they do not override matrix locking.

Optional: pull curated sites from RegulonDB (network access required):

```bash
cruncher fetch sites --tf lexA --tf cpxR --update -c "$CONFIG"
```

## Lock + parse

```bash
cruncher lock  -c "$CONFIG"
cruncher parse -c "$CONFIG"
```

`lock` pins the exact PWM artifacts for reproducible runs. `parse` validates the locked motifs and writes a parse manifest used by sampling.
If `.cruncher/parse` already exists from a prior run, re-run parse with `--force-overwrite`.

## Sample + analyze

```bash
cruncher sample  -c "$CONFIG" && \
cruncher analyze -c "$CONFIG" && \
cruncher analyze --summary -c "$CONFIG"
```

If `outputs/` already exists from a prior run, re-run sample with `--force-overwrite`.
If sample fails with an elite-filter message, relax `sample.elites.filter.min_per_tf_norm`, increase
`sample.sequence_length`, or increase `sample.budget.draws`, then re-run sample.

## Fast rerun after config edits

After editing `config.yaml`, this path re-locks and regenerates analysis without
manual cleanup:

```bash
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
CONFIG="$PWD/config.yaml"

pixi run cruncher -- lock -c "$CONFIG"
pixi run cruncher -- parse --force-overwrite -c "$CONFIG"
pixi run cruncher -- sample --force-overwrite -c "$CONFIG"
pixi run cruncher -- analyze --summary -c "$CONFIG"
```

## Inspect results

Run artifacts live under:

```
<workspace>/outputs/
```

Key files:

- `analysis/summary.json`
- `analysis/report.md`
- `analysis/report.json`
- curated plots in `plots/`: `plot__opt_trajectory.*`,
  `plot__opt_trajectory_sweep.*`, `plot__elites_nn_distance.*`, `plot__overlap_panel.*`
  (and `plot__health_panel.*` if a trace is present; `plot__overlap_panel.*` is skipped when `n_elites < 2`)
- analysis tables in `analysis/` use `table__*` filenames (for example
  `table__scores_summary.parquet` and `table__metrics_joint.parquet`)
- elite hit metadata: `optimize/elites_hits.parquet`
- random baseline cloud: `optimize/random_baseline.parquet`
- random baseline hits: `optimize/random_baseline_hits.parquet`

## Optional: motif discovery

If you want to rebuild PWMs from cached sites, enable discovery in the config and run:

```bash
cruncher discover motifs --tf lexA --tf cpxR -c "$CONFIG"
cruncher lock -c "$CONFIG"
cruncher parse -c "$CONFIG"
```

Discovery writes new motif matrices into the catalog and requires MEME Suite.

## Related docs

- [Config reference](../reference/config.md)
- [Intent + lifecycle](../guides/intent_and_lifecycle.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
