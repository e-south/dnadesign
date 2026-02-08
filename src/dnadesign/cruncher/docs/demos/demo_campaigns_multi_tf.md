# Campaign demo (multi‑TF)

## Contents
- [Overview](#overview)
- [Demo setup](#demo-setup)
- [Cache sites](#cache-sites)
- [Generate a campaign](#generate-a-campaign)
- [Lock + parse](#lock--parse)
- [Sample + analyze](#sample--analyze)
- [Inspect campaign summary](#inspect-campaign-summary)
- [Related docs](#related-docs)

## Overview

Campaigns expand regulator categories into explicit regulator sets. This demo shows how to generate a campaign, run sampling, and summarize results across runs.
The demo config enables adaptive move/proposal tuning with strict PT adaptation checks so campaign runs fail fast when PT tuning is unhealthy.

## Demo setup

```bash
cd src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf
CONFIG="$PWD/config.yaml"
cruncher() { pixi run cruncher -- "$@"; }
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

## Generate a campaign

```bash
cruncher campaign generate --campaign demo_pair --out campaign_demo_pair.yaml -c "$CONFIG"
DERIVED="$PWD/campaign_demo_pair.yaml"
```

This writes a derived config that contains explicit `workspace.regulator_sets` and a `campaign` metadata block. Use that derived config for runs.

## Lock + parse

```bash
cruncher lock  -c "$DERIVED"
cruncher parse -c "$DERIVED"
```

## Sample + analyze

```bash
cruncher sample  -c "$DERIVED"
cruncher analyze --summary -c "$DERIVED"
```

## Inspect campaign summary

```bash
cruncher campaign summarize --campaign demo_pair -c "$DERIVED"
```

`campaign summarize` auto-repairs stale sample run-index entries (for example after manual output cleanup) and logs what it removed.
If you want to repair explicitly first:

```bash
cruncher runs repair-index --apply -c "$DERIVED"
```

Summary artifacts live under:

```
<workspace>/outputs/campaign/<campaign_name>/latest/
  output/campaign_summary.csv
  output/campaign_best.csv
  output/campaign_manifest.json
  plots/plot__*.png
```

## Related docs

- [Config reference](../reference/config.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
