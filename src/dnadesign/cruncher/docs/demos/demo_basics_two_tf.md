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
- [Export DenseGen Motifs](#export-densegen-motifs)
- [Sample + analyze](#sample--analyze)
- [Export sequence tables](#export-sequence-tables)
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
The bundled config uses targeted insertion proposals with a six-chain Gibbs annealing optimizer (`optimizer.kind=gibbs_anneal`, `optimizer.chains=6`) plus a colder piecewise cooling tail (`cooling.kind=piecewise`, final `beta=14.0`) and linear move scheduling toward less Gibbs-dominant late sweeps. Adaptive move weights are disabled; proposal scaling remains enabled with a cold-tail freeze (`proposal_adapt.freeze_after_beta=9.0`). Gibbs inertia is enabled (`gibbs_inertia.p_stay_end=0.75`) to reduce late single-site flip jitter. The softmin schedule remains fixed (`beta_end=12.0`), and sweep plots default to best-so-far (`analysis.trajectory_sweep_mode=best_so_far`).

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
pixi run cruncher -- fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"
pixi run cruncher -- discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c "$CONFIG"
pixi run cruncher -- lock -c "$CONFIG"
pixi run cruncher -- parse --force-overwrite -c "$CONFIG"
pixi run cruncher -- sample --force-overwrite -c "$CONFIG"
pixi run cruncher -- analyze --summary -c "$CONFIG"
pixi run cruncher -- export sequences --latest -c "$CONFIG"
pixi run cruncher -- catalog logos --source demo_merged_meme_oops --tf lexA --tf cpxR -c "$CONFIG"
find outputs -type f -path '*/plots/analysis/*' | sort
find outputs -type f -path '*/export/sequences/*' | sort
find outputs -type f -path '*/plots/logos/*' -name '*.png' | sort
```

If `sample` fails to produce enough elites, increase `sample.budget.draws`, set `sample.elites.select.pool_size: all`,
or lower `sample.elites.select.diversity`, then rerun.
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

Fetch both sources before discovery so the TFBS pool is explicit and reproducible.

```bash
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"
```

`catalog.combine_sites=true` in this demo, so discovery uses all cached site entries per TF (across both sources).
The local MEME files provide DAP-seq sites; RegulonDB adds curated sites.

## Discover MEME OOPS motifs

Run MEME in OOPS mode over the merged site sets and write discovered motifs under
source `demo_merged_meme_oops`.

Preflight once per machine:

```bash
cruncher discover check
```

This demo config sets `discover.tool_path` to the repo-local Pixi MEME bin
(`.pixi/envs/default/bin`), so discovery/analyze should run without setting
`MEME_BIN`. If your MEME install lives elsewhere, update `discover.tool_path`
or set `MEME_BIN` before running discovery.

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

## Export DenseGen motifs

Export the same discovered motifs used by lock/parse/sample/analyze:

```bash
cruncher catalog export-densegen --set 1 --out outputs/densegen/pwms -c "$CONFIG"
```

This keeps DenseGen exports on the same motif provenance path as optimization and showcase plotting.

## Sample + analyze

Intent:
- `sample` generates optimization artifacts from locked motifs.
- `analyze` converts those artifacts into curated plots/tables/reports.
- `catalog logos` renders motif logos tied to the same discovered source.

```bash
cruncher sample -c "$CONFIG"
cruncher analyze -c "$CONFIG"
cruncher analyze --summary -c "$CONFIG"
cruncher catalog logos --source demo_merged_meme_oops --tf lexA --tf cpxR -c "$CONFIG"
```

During sampling, logs now print `Sampling PWM width <TF>: source=<...> effective=<...> ... action=<trimmed|unchanged>`
so it is explicit when `sample.motif_width` changed widths versus leaving discovered widths unchanged.

If `outputs/` already exists from a prior run, re-run sample with `--force-overwrite`.
If sample fails with an elite-count message, increase `sample.budget.draws`, set
`sample.elites.select.pool_size: all`, or lower `sample.elites.select.diversity`, then re-run sample.

## Export sequence tables

Intent:
- Emit wrapper-friendly sequence contracts from the analyzed sample run.
- Keep downstream integrations independent from internal optimize table shapes.

```bash
cruncher export sequences --latest -c "$CONFIG"
```

This writes:
- `outputs/export/sequences/table__monospecific_consensus_sites.parquet`
- `outputs/export/sequences/table__monospecific_elite_windows.parquet`
- `outputs/export/sequences/table__bispecific_elite_windows.parquet`
- `outputs/export/sequences/table__multispecific_elite_windows.parquet` (empty for 2-TF runs)
- `outputs/export/sequences/export_manifest.json`

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
pixi run cruncher -- export sequences --latest -c "$CONFIG"
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

- run metadata: `run/run_manifest.json`, `run/run_status.json`, `run/config_used.yaml`
- pinned inputs: `provenance/lockfile.json`, `provenance/parse_manifest.json`
- optimizer tables: `optimize/tables/sequences.parquet`, `optimize/tables/elites.parquet`,
  `optimize/tables/elites_hits.parquet`, `optimize/tables/random_baseline.parquet`,
  `optimize/tables/random_baseline_hits.parquet`
- optimizer state: `optimize/state/trace.nc`, `optimize/state/metrics.jsonl`, `optimize/state/elites.yaml`
- analysis reports: `analysis/reports/summary.json`, `analysis/reports/report.json`, `analysis/reports/report.md`
- analysis manifests: `analysis/manifests/manifest.json`, `analysis/manifests/plot_manifest.json`,
  `analysis/manifests/table_manifest.json`
- analysis plots: `plots/analysis/chain_trajectory_scatter.*`, `plots/analysis/chain_trajectory_sweep.*`,
  `plots/analysis/elites_nn_distance.*`, `plots/analysis/elites_showcase.*` (and `plots/analysis/health_panel.*` when trace is present)
- analysis tables: `analysis/tables/table__scores_summary.parquet`, `analysis/tables/table__metrics_joint.parquet`
- sequence exports: `export/sequences/table__*.parquet`, `export/sequences/export_manifest.json`
- motif logos: `plots/logos/*.png`

## Related docs

- [Config reference](../reference/config.md)
- [Intent + lifecycle](../guides/intent_and_lifecycle.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
