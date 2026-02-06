# Campaign demo (multi‑TF)

## Contents
- [Overview](#overview)
- [Demo setup](#demo-setup)
- [Generate a campaign](#generate-a-campaign)
- [Lock + parse](#lock--parse)
- [Sample + analyze](#sample--analyze)
- [Inspect campaign summary](#inspect-campaign-summary)
- [Related docs](#related-docs)

## Overview

Campaigns expand regulator categories into explicit regulator sets. This demo shows how to generate a campaign, run sampling, and summarize results across runs.

## Demo setup

```bash
cd src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf
CONFIG="$PWD/config.yaml"
cruncher() { pixi run cruncher -- "$@"; }
```

## Generate a campaign

```bash
cruncher campaign generate --campaign regulators_v1 -c "$CONFIG"
```

This writes a derived config that contains explicit `workspace.regulator_sets` and a `campaign` metadata block. Use that derived config for runs.

## Lock + parse

```bash
cruncher lock  -c "$CONFIG"
cruncher parse -c "$CONFIG"
```

## Sample + analyze

```bash
cruncher sample  -c "$CONFIG"
cruncher analyze -c "$CONFIG"
```

## Inspect campaign summary

```bash
cruncher campaign summarize --campaign regulators_v1 -c "$CONFIG"
```

Summary artifacts live under:

```
<workspace>/outputs/campaigns/<campaign_id>/
```

## Related docs

- [Config reference](../reference/config.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
