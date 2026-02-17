# Densegen Prep Demo (Three-TF)

## Contents
- [Overview](#overview)
- [Demo setup](#demo-setup)
- [Reset demo](#reset-demo)
- [Cache sites by source](#cache-sites-by-source)
- [Discover merged motifs (MEME OOPS)](#discover-merged-motifs-meme-oops)
- [Lock + parse](#lock--parse)
- [Export DenseGen Motifs](#export-densegen-motifs)
- [Sample + analyze](#sample--analyze)
- [Export sequence tables](#export-sequence-tables)
- [Inspect results](#inspect-results)
- [Related docs](#related-docs)

## Overview

This workspace prepares a three-TF run (`lexA`, `cpxR`, `baeR`) using a strict
two-stage contract:
1. merge cached site sets per TF, then run MEME in OOPS mode;
2. lock/sample/analyze against discovered motifs only.

`catalog.source_preference` is pinned to the discovered source
`demo_merged_meme_oops_three_tf`, so `lock` fails until discovery has been run.
This prevents accidental fallback to raw source motifs/sites and keeps optimizer +
analysis motif provenance explicit.

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
- `lexA`, `cpxR` from `demo_local_meme` + `regulondb`
- `baeR` from `baer_chip_exo` + `regulondb`

```bash
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source baer_chip_exo --tf baeR --update -c "$CONFIG"
cruncher fetch sites --source regulondb      --tf baeR --update -c "$CONFIG"
```

Using the same lexA/cpxR source mix as the two-TF demo keeps discovered motif widths
aligned across workspaces.

## Discover merged motifs (MEME OOPS)

`catalog.combine_sites=true` merges all cached site entries per TF before discovery.
`discover motifs` writes discovered matrices to
`demo_merged_meme_oops_three_tf`, which is the only source used by lock/sample/analyze.

```bash
cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id demo_merged_meme_oops_three_tf -c "$CONFIG"
```

## Lock + parse

```bash
cruncher lock  -c "$CONFIG"
cruncher parse -c "$CONFIG"
```

If `.cruncher/parse` already exists from a prior run, re-run parse with `--force-overwrite`.
If `lock` fails, re-run discovery first (this demo intentionally does not fall back
to raw sources).

## Export DenseGen Motifs

Export the same discovered motifs used by optimization and analysis:

```bash
cruncher catalog export-densegen --set 1 --out outputs/densegen/pwms -c "$CONFIG"
```

## Sample + analyze

Intent:
- `sample` optimizes sequences against the locked three-TF motif set.
- `analyze` produces reports and plots from the generated sample artifacts.

```bash
cruncher sample -c "$CONFIG"
cruncher analyze --summary -c "$CONFIG"
```

If `outputs/` already exists from a prior run, re-run sample with `--force-overwrite`.

## Export sequence tables

Intent:
- Emit stable downstream sequence contracts for wrappers/operators.

```bash
cruncher export sequences --latest -c "$CONFIG"
```

## Inspect results

Outputs are written under:

```
<workspace>/outputs/
```

Key files:
- `run/run_manifest.json`, `run/run_status.json`, `run/config_used.yaml`
- `analysis/reports/summary.json`, `analysis/reports/report.md`
- `plots/analysis/elites_showcase.pdf` (and other analysis plots)
- `analysis/tables/table__scores_summary.parquet` (and other analysis tables)
- `export/sequences/table__*.parquet`, `export/sequences/export_manifest.json`
- `plots/logos/*.png`

## Related docs

- [Config reference](../reference/config.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
