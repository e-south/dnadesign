# Campaign demo (multi‑TF)

## Contents
- [Overview](#overview)
- [Demo setup](#demo-setup)
- [Reset demo](#reset-demo)
- [Cache sites](#cache-sites)
- [Run campaign directly](#run-campaign-directly)
- [Optional: materialize expansion](#optional-materialize-expansion)
- [Inspect campaign summary](#inspect-campaign-summary)
- [Related docs](#related-docs)

## Overview

Campaigns expand regulator categories into explicit regulator sets. This demo runs a campaign directly with `--campaign` (no derived config required), then summarizes results across runs.
The demo config uses a ten-chain Gibbs annealing optimizer (`optimizer.kind=gibbs_anneal`, `optimizer.chains=10`) with a colder piecewise schedule (`cooling.kind=piecewise`, final `beta=18.0`) and linear move scheduling toward less Gibbs-dominant tail behavior. Adaptive move weights are disabled and proposal scaling remains explicit under `sample.moves.overrides.proposal_adapt`, with a cold-tail freeze (`freeze_after_beta=8.0`). Gibbs inertia is enabled (`gibbs_inertia.p_stay_end=0.70`) to damp late single-site jitter, and sweep plots default to best-so-far (`analysis.trajectory_sweep_mode=best_so_far`).

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

## Cache sites

The multi-TF demo is configured for site-derived PWMs (`catalog.pwm_source: sites`).
For `demo_pair`, only `lexA` and `cpxR` are required:

```bash
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
```

Optional: prefetch additional TFs used by `demo_categories` / `demo_categories_best`:

```bash
cruncher fetch sites --source demo_local_meme --tf acrR --tf lrp --tf rcdA --tf soxR --update -c "$CONFIG"
cruncher fetch sites --source regulondb --tf baeR --tf fnr --tf fur --tf soxS --update -c "$CONFIG"
```

## Run campaign directly

```bash
cruncher lock    --campaign demo_pair -c "$CONFIG"
cruncher parse   --campaign demo_pair -c "$CONFIG"
cruncher sample  --campaign demo_pair -c "$CONFIG"
cruncher analyze --campaign demo_pair --summary -c "$CONFIG"
```

If `.cruncher/parse` or `outputs/` already exist from a prior run, add
`--force-overwrite` to `parse` and `sample`.

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
  plots/plot__*.png
```

Campaign summary plots are currently always emitted as PNG (independent of `analysis.plot_format`).

## Related docs

- [Config reference](../reference/config.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
