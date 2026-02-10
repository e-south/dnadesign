# Two-TF demo (end-to-end)

## Contents
- [Overview](#overview)
- [TL;DR (copy/paste e2e)](#tldr-copypaste-e2e)
- [Demo setup](#demo-setup)
- [Reset demo](#reset-demo)
- [Cache Binding Sites (DAP-seq + RegulonDB)](#cache-binding-sites-dap-seq--regulondb)
- [Discover MEME OOPS Motifs](#discover-meme-oops-motifs)
- [Lock + parse](#lock--parse)
- [Render MEME Logos](#render-meme-logos)
- [Sample + analyze](#sample--analyze)
- [Fast rerun after config edits](#fast-rerun-after-config-edits)
- [Inspect results](#inspect-results)
- [Related docs](#related-docs)

## Overview

This demo designs fixed-length sequences that satisfy two PWMs (LexA + CpxR). It follows the full lifecycle:

1. cache binding sites
2. discover motifs
3. lock
4. parse
5. sample
6. analyze

Cruncher scores each TF by the best PWM match anywhere in the sequence on either strand (when `objective.bidirectional=true`). It optimizes the weakest TF by default (`objective.combine=min`) and selects diverse elites via TFBS-core MMR.
The demo builds MEME OOPS motifs from merged DAP-seq (`demo_local_meme`) + RegulonDB sites, then locks sampling to those discovered motifs. Sequence length is fixed at 16 bp, and sampling enforces `sample.motif_width.maxw=16` via max-information contiguous windowing.
The bundled config uses targeted insertion proposals plus adaptive move weights/proposal sizing with strict PT adaptation (`n_temps=3`, `temp_max=10`, `swap_stride=8`) and a fixed softmin schedule (`beta_end=6`) for a less jumpy optimization trajectory.

For the full intent, lifecycle, and config mapping, see [Intent + lifecycle](../guides/intent_and_lifecycle.md).

## TL;DR (copy/paste e2e)

Use this block for a clean, repeatable end-to-end run that regenerates plots.

```bash
set -euo pipefail
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
CONFIG="$PWD/config.yaml"
rm -rf outputs
rm -rf .cruncher/parse .cruncher/locks .cruncher/campaigns
rm -f .cruncher/run_index.json
pixi run cruncher -- fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
pixi run cruncher -- fetch sites --source regulondb --tf lexA --tf cpxR --update -c "$CONFIG"
pixi run cruncher -- discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c "$CONFIG"
pixi run cruncher -- lock -c "$CONFIG"
pixi run cruncher -- parse --force-overwrite -c "$CONFIG"
pixi run cruncher -- sample --force-overwrite -c "$CONFIG"
pixi run cruncher -- analyze --summary -c "$CONFIG"
pixi run cruncher -- catalog logos --source demo_merged_meme_oops --tf lexA --tf cpxR -c "$CONFIG"
find outputs -type f -name 'plot__*.png' | sort
find outputs -type f -path '*/logos/*' -name '*.png' | sort
```

If `sample` fails due elite filters, adjust `sample.elites.filter.min_per_tf_norm`,
`sample.sequence_length`, or `sample.budget.draws`, then rerun.
If `lock` fails after reset, run `discover motifs` first; this demo intentionally locks only to discovered motifs.

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

## Cache Binding Sites (DAP-seq + RegulonDB)

Fetch binding sites from both sources so each TF has a larger merged site pool.

```bash
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"
```

`catalog.combine_sites=true` in this demo, so discovery uses all cached site entries per TF (across both sources).
The local MEME files provide DAP-seq sites; RegulonDB adds curated sites.

## Discover MEME OOPS motifs

Run MEME in OOPS mode over the merged site sets and write discovered motifs under
source `demo_merged_meme_oops`.

```bash
cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c "$CONFIG"
```

This demo leaves MEME width unconstrained at discovery time (`discover.minw/maxw: null`) so logos and catalog inspection keep full discovered motifs.
Width enforcement is applied only in sampling via `sample.motif_width.maxw=16`.
`discover motifs` prints `Tool width` and `Width bounds`; `tool_default` means no discovery-time width flags were passed to MEME/STREME.

## Lock + parse

```bash
cruncher lock  -c "$CONFIG"
cruncher parse -c "$CONFIG"
```

`lock` pins exact discovered motif IDs/hashes for reproducible runs. `parse` validates those locked motifs and writes a parse manifest used by sampling.
If `.cruncher/parse` already exists from a prior run, re-run parse with `--force-overwrite`.
This demo pins `catalog.source_preference` to `demo_merged_meme_oops` to prevent accidental fallback to local 21/22 bp motifs.

## Render MEME logos

Render logos from the discovered MEME motifs as a quick validation that the cached motif shapes look sane:

```bash
cruncher catalog logos --source demo_merged_meme_oops --tf lexA --tf cpxR -c "$CONFIG"
```

When you run `sample --force-overwrite`, generate logos after sampling so they are part of the final output snapshot.

## Sample + analyze

```bash
cruncher sample  -c "$CONFIG" && \
cruncher analyze -c "$CONFIG" && \
cruncher analyze --summary -c "$CONFIG" && \
cruncher catalog logos --source demo_merged_meme_oops --tf lexA --tf cpxR -c "$CONFIG"
```

During sampling, logs now print `Sampling PWM width <TF>: source=<...> effective=<...> ... action=<trimmed|unchanged>`
so it is explicit when `sample.motif_width` changed widths versus leaving discovered widths unchanged.

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
pixi run cruncher -- catalog logos --source demo_merged_meme_oops --tf lexA --tf cpxR -c "$CONFIG"
```

If you changed site/discovery settings, refresh discovery before lock:

```bash
pixi run cruncher -- fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
pixi run cruncher -- fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"
pixi run cruncher -- discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c "$CONFIG"
pixi run cruncher -- lock -c "$CONFIG"
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
- motif logos under `logos/catalog/<run_name>/`
- curated plots in `plots/`: `plot__opt_trajectory.*`,
  `plot__opt_trajectory_sweep.*`, `plot__elites_nn_distance.*`, `plot__overlap_panel.*`
  (and `plot__health_panel.*` if a trace is present; `plot__overlap_panel.*` is skipped when `n_elites < 2`)
- analysis tables in `analysis/` use `table__*` filenames (for example
  `table__scores_summary.parquet` and `table__metrics_joint.parquet`)
- elite hit metadata: `optimize/elites_hits.parquet`
- random baseline cloud: `optimize/random_baseline.parquet`
- random baseline hits: `optimize/random_baseline_hits.parquet`

## Related docs

- [Config reference](../reference/config.md)
- [Intent + lifecycle](../guides/intent_and_lifecycle.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
