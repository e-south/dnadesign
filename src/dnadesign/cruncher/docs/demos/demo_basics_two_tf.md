# Two‑TF demo (end‑to‑end)

## Contents
- [Overview](#overview)
- [Demo setup](#demo-setup)
- [Cache motifs and sites](#cache-motifs-and-sites)
- [Lock + parse](#lock--parse)
- [Sample + analyze](#sample--analyze)
- [Inspect results](#inspect-results)
- [Optional: motif discovery](#optional-motif-discovery)
- [Related docs](#related-docs)

## Overview

This demo designs fixed‑length sequences that satisfy two PWMs (LexA + CpxR). It follows the full lifecycle:

1. cache inputs
2. lock
3. parse
4. sample
5. analyze

Cruncher scores each TF by the best PWM match anywhere in the sequence on either strand (when `objective.bidirectional=true`). It optimizes the weakest TF by default (`objective.combine=min`) and selects diverse elites via TFBS‑core MMR.

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

## Cache motifs and sites

The demo ships with local MEME motifs and MEME BLOCKS sites.

```bash
cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites  --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
```

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

## Sample + analyze

```bash
cruncher sample  -c "$CONFIG"
cruncher analyze -c "$CONFIG"
cruncher analyze --summary -c "$CONFIG"
```

## Inspect results

Run artifacts live under:

```
<workspace>/outputs/sample/<run_name>/
```

Key files:

- `analysis/summary.json`
- `analysis/report.md`
- `analysis/report.json`
- curated plots: `analysis/plot__run__summary.*`, `analysis/plot__opt__trajectory.*`, `analysis/plot__elites__nn_distance.*`,
  `analysis/plot__overlap__panel.*` (and `analysis/plot__health__panel.*` if a trace is present)
- elite hit metadata: `artifacts/elites_hits.parquet`
- random baseline cloud: `artifacts/random_baseline.parquet`

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
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
