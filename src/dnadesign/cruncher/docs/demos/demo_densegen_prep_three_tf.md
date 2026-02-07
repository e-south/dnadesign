# Densegen Prep Demo (Three-TF)

## Contents
- [Overview](#overview)
- [Demo setup](#demo-setup)
- [Cache sites by source](#cache-sites-by-source)
- [Lock + parse](#lock--parse)
- [Sample + analyze](#sample--analyze)
- [Related docs](#related-docs)

## Overview

This workspace prepares a three-TF run (`lexA`, `cpxR`, `baeR`) using site-derived PWMs.
It is a practical prep flow for downstream densegen exports and validates that mixed site
sources can be locked, parsed, sampled, and analyzed end-to-end.

## Demo setup

```bash
cd src/dnadesign/cruncher/workspaces/densegen_prep_three_tf
CONFIG="$PWD/config.yaml"
cruncher() { pixi run cruncher -- "$@"; }
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

## Sample + analyze

```bash
cruncher sample  -c "$CONFIG"
cruncher analyze --summary -c "$CONFIG"
```

Plots and tables are written under:

```
<workspace>/outputs/sample/latest/
```

## Related docs

- [Config reference](../reference/config.md)
- [Sampling + analysis](../guides/sampling_and_analysis.md)
- [CLI reference](../reference/cli.md)
