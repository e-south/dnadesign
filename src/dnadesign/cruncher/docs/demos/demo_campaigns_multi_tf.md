# Campaign demo (multi‑TF)

## Contents
- [Overview](#overview)
- [Demo setup](#demo-setup)
- [Reset demo](#reset-demo)
- [Cache merged TFBS sources](#cache-merged-tfbs-sources)
- [Discover merged motifs (MEME OOPS)](#discover-merged-motifs-meme-oops)
- [Run campaign directly](#run-campaign-directly)
- [Export DenseGen motifs](#export-densegen-motifs)
- [Export sequence tables](#export-sequence-tables)
- [Optional: materialize expansion](#optional-materialize-expansion)
- [Inspect campaign summary](#inspect-campaign-summary)
- [Related docs](#related-docs)

## Overview

This demo uses the same strict provenance contract as the two-TF and three-TF demos:

1. cache TFBS sets from available sources
2. merge TFBS sets per TF
3. run MEME in OOPS mode on merged sites
4. lock/parse/sample/analyze against discovered motifs only

The config pins `catalog.source_preference` to `demo_merged_meme_oops_campaign`, so
`lock` fails until discovery has been run. There is no silent fallback to raw local
or RegulonDB motifs during optimization.

## Demo setup

```bash
cd src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf
CONFIG="$PWD/config.yaml"
cruncher() { pixi run cruncher -- "$@"; }
```

## Reset demo

Run this before repeating the campaign e2e flow.
It removes run artifacts, campaign summary outputs, and stale generated campaign files in the workspace root.

```bash
cd src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf
rm -rf outputs
rm -rf .cruncher/parse .cruncher/locks .cruncher/campaigns
rm -f .cruncher/run_index.json
find . -maxdepth 1 -type f \( -name 'campaign_*.yaml' -o -name 'campaign_*.campaign_manifest.json' \) -delete
```

## Cache merged TFBS sources

### Required for `demo_pair` (`lexA`, `cpxR`)

```bash
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"
```

### Required when `baeR` is present in a campaign set

```bash
cruncher fetch sites --source baer_chip_exo --tf baeR --update -c "$CONFIG"
cruncher fetch sites --source regulondb    --tf baeR --update -c "$CONFIG"
```

### Optional prefetch for `demo_categories` / `demo_categories_best`

DAP-seq is available in this workspace for: `acrR`, `cpxR`, `lexA`, `lrp`, `rcdA`, `soxR`.
For those TFs, cache both local DAP-seq and RegulonDB when available:

```bash
cruncher fetch sites --source demo_local_meme --tf acrR --tf lrp --tf rcdA --tf soxR --update -c "$CONFIG"
cruncher fetch sites --source regulondb      --tf acrR --tf lrp --tf rcdA --tf soxR --update -c "$CONFIG"
```

For TFs without local DAP-seq in this demo (`fnr`, `fur`, `soxS`), use RegulonDB:

```bash
cruncher fetch sites --source regulondb --tf fnr --tf fur --tf soxS --update -c "$CONFIG"
```

## Discover merged motifs (MEME OOPS)

Discover motifs into the pinned campaign source:

```bash
cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id demo_merged_meme_oops_campaign -c "$CONFIG"
```

For campaign sets that include additional TFs, include them in discovery too (example):

```bash
cruncher discover motifs --tf lexA --tf cpxR --tf baeR --tf acrR --tf lrp --tf rcdA --tf soxR --tf fnr --tf fur --tf soxS --tool meme --meme-mod oops --source-id demo_merged_meme_oops_campaign -c "$CONFIG"
```

## Run campaign directly

Intent:
- `lock` pins discovered motifs for the campaign TF set.
- `parse` validates pinned motifs and prepares parse artifacts.
- `sample` runs optimization.
- `analyze` renders reports/plots for the sampled run.

```bash
cruncher lock    --campaign demo_pair -c "$CONFIG"
cruncher parse   --campaign demo_pair -c "$CONFIG"
cruncher sample  --campaign demo_pair -c "$CONFIG"
cruncher analyze --campaign demo_pair --summary -c "$CONFIG"
```

If `.cruncher/parse` or `outputs/` already exist from a prior run, add
`--force-overwrite` to `parse` and `sample`.

## Export DenseGen motifs

Export discovered motifs for the same campaign TF set:

```bash
cruncher catalog export-densegen --tf lexA --tf cpxR --out outputs/densegen/pwms -c "$CONFIG"
```

For larger campaign TF sets, pass the full TF list explicitly with repeated `--tf`.

## Export sequence tables

Export downstream sequence contracts from the latest sampled campaign run:

```bash
cruncher export sequences --latest -c "$CONFIG"
```

This writes run-level contract tables under:

```
<workspace>/outputs/export/sequences/
```

including:
- `table__monospecific_consensus_sites.parquet`
- `table__monospecific_elite_windows.parquet`
- `table__bispecific_elite_windows.parquet`
- `table__multispecific_elite_windows.parquet`
- `export_manifest.json`

## Optional: materialize expansion

`campaign generate` is optional and writes under workspace state by default:

```bash
cruncher campaign generate --campaign demo_pair -c "$CONFIG"
```

Default outputs:

```
<workspace>/.cruncher/campaigns/<campaign>/generated.yaml
<workspace>/.cruncher/campaigns/<campaign>/generated.campaign_manifest.json
```

## Inspect campaign summary

```bash
cruncher campaign summarize --campaign demo_pair -c "$CONFIG"
```

`campaign summarize` auto-repairs stale sample run-index entries (for example after manual output cleanup) and logs what it removed.
If you want to repair explicitly first:

```bash
cruncher runs repair-index --apply -c "$CONFIG"
```

Summary artifacts live under:

```
<workspace>/outputs/campaign/<campaign_name>/
  analysis/campaign_summary.csv
  analysis/campaign_best.csv
  analysis/campaign_manifest.json
  plots/*.png
```

Campaign summary plots are currently always emitted as PNG (independent of `analysis.plot_format`).

## Related docs

- [Config reference](../reference/config.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
