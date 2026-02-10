# Densegen Prep Demo (Three-TF)

## Contents
- [Overview](#overview)
- [Demo setup](#demo-setup)
- [Reset demo](#reset-demo)
- [Cache sites by source](#cache-sites-by-source)
- [Lock + parse](#lock--parse)
- [Sample + analyze](#sample--analyze)
- [Related docs](#related-docs)

## Overview

This workspace prepares a three-TF run (`lexA`, `cpxR`, `baeR`) using site-derived PWMs.
It is a practical prep flow for downstream densegen exports and validates that mixed site
sources can be locked, parsed, sampled, and analyzed end-to-end. The bundled
config uses Gibbs annealing with piecewise cooling, explicit proposal adaptation
freeze in the cold tail, and Gibbs inertia for late-stage stability; sweep plots
default to best-so-far (`analysis.trajectory_sweep_mode=best_so_far`).

## Demo setup

```bash
cd src/dnadesign/cruncher/workspaces/densegen_prep_three_tf
CONFIG="$PWD/config.yaml"
cruncher() { pixi run cruncher -- "$@"; }
```

## Reset demo

Run this before re-running the full three-TF flow.
It clears run artifacts and workspace state while leaving shared catalog caches intact.

```bash
cd src/dnadesign/cruncher/workspaces/densegen_prep_three_tf
rm -rf outputs
rm -rf .cruncher/parse .cruncher/locks .cruncher/campaigns
rm -f .cruncher/run_index.json
```

## Cache sites by source

This demo intentionally uses multiple sources:
- `lexA`, `cpxR` from `demo_local_meme`
- `baeR` from `baer_chip_exo`

```bash
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source baer_chip_exo --tf baeR --update -c "$CONFIG"
```

## Lock + parse

```bash
cruncher lock  -c "$CONFIG"
cruncher parse -c "$CONFIG"
```

If `.cruncher/parse` already exists from a prior run, re-run parse with `--force-overwrite`.

## Sample + analyze

```bash
cruncher sample  -c "$CONFIG"
cruncher analyze --summary -c "$CONFIG"
```

If `outputs/` already exists from a prior run, re-run sample with `--force-overwrite`.

Plots and tables are written under:

```
<workspace>/outputs/
```

## Related docs

- [Config reference](../reference/config.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
