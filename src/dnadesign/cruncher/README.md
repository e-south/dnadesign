# Cruncher

Cruncher designs short, fixed-length DNA sequences that jointly satisfy one or more TF motifs, then returns a diverse elite set.

## Contents

1. [Overview](#overview)
2. [Quickstart](#quickstart)
3. [Documentation map](#documentation-map)

## Overview

Cruncher is an optimization tool with strict contracts:

1. Cache TF binding sites per source.
2. Merge sites per TF and discover motifs (typically MEME OOPS).
3. Lock discovered motif artifacts for reproducible sampling.
4. Parse, sample, and analyze from locked inputs.

Core properties:
- fixed-length sequence optimization (`sample.sequence_length`)
- best-hit PWM scoring per TF (optional bidirectional scan)
- elite diversity via TFBS-core MMR
- fail-fast behavior for invalid config/artifact states (no silent fallback)

For the complete intent/lifecycle model, see `docs/guides/intent_and_lifecycle.md`.

## Quickstart

```bash
cd src/dnadesign/cruncher/workspaces/demo_basics_two_tf
CONFIG="$PWD/config.yaml"
cruncher() { pixi run cruncher -- "$@"; }

# 1) Cache TFBS from both sources used by this demo.
cruncher fetch sites --source demo_local_meme --tf lexA --tf cpxR --update -c "$CONFIG"
cruncher fetch sites --source regulondb      --tf lexA --tf cpxR --update -c "$CONFIG"

# 2) Discover merged motifs (MEME OOPS).
cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --source-id demo_merged_meme_oops -c "$CONFIG"

# 3) Lock + parse + sample + analyze.
cruncher lock -c "$CONFIG"
cruncher parse --force-overwrite -c "$CONFIG"
cruncher sample --force-overwrite -c "$CONFIG"
cruncher analyze --summary -c "$CONFIG"
cruncher export sequences --latest -c "$CONFIG"
```

Notes:
- Analysis entrypoints are `outputs/analysis/summary.json` and `outputs/analysis/report.md`.
- Sequence-export entrypoints are under `outputs/export/sequences/`.
- Elites showcase rendering goes through the baserender public API (`dnadesign.baserender`) only.
- For demo-specific source mix and campaign flows, use the demo docs below.

## Documentation map

1. [Docs index](docs/index.md)
2. [Two-TF demo](docs/demos/demo_basics_two_tf.md)
3. [Campaign demo (multi-TF)](docs/demos/demo_campaigns_multi_tf.md)
4. [Three-TF DenseGen prep demo](docs/demos/demo_densegen_prep_three_tf.md)
5. [Ingestion guide](docs/guides/ingestion.md)
6. [MEME Suite guide](docs/guides/meme_suite.md)
7. [Sampling and analysis](docs/guides/sampling_and_analysis.md)
8. [Config reference](docs/reference/config.md)
9. [CLI reference](docs/reference/cli.md)
10. [Architecture reference](docs/reference/architecture.md)
