## Study: Sequence Length vs Score (Two-TF Demo)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23


### Contents
- [What this Study measures](#what-this-study-measures)
- [Run it](#run-it)
- [Read the outputs](#read-the-outputs)
- [Clean old Study outputs](#clean-old-study-outputs)

This guide uses the `demo_pairwise` workspace Study spec:

- `workspaces/demo_pairwise/configs/studies/length_vs_score.study.yaml`

### What this Study measures

This Study runs a sampling sweep across the workspace length-study range (`15..50` for most workspaces; `19..50` for constrained
workspaces like `pairwise_laci_arac`) with replicate seeds.
It answers: how does score/diversity behavior change as length increases?

The sweep is sampling-first:

- each length is a full sampling trial
- aggregate tables/plots summarize across replicate seeds
- non-swept sampling behavior is inherited from `configs/config.yaml`

### Run it

Prerequisite:

- the workspace must already have lock/parse-ready motif data (for `demo_pairwise`, run the two-TF demo data-prep sequence first, including merged motif discovery into `demo_merged_meme_oops`).

```bash
set -euo pipefail
cd src/dnadesign/cruncher/workspaces/demo_pairwise

uv run cruncher lock configs/config.yaml
uv run cruncher parse configs/config.yaml
uv run cruncher study run --spec configs/studies/length_vs_score.study.yaml
uv run cruncher study list
```

Find the generated run directory:

```bash
uv run cruncher study show --run outputs/studies/length_vs_score/<study_id>
open outputs/plots/study__length_vs_score__<study_id>__plot__sequence_length_tradeoff.pdf
```

### Read the outputs

Primary artifacts:

- `tables/table__trial_metrics.parquet`
- `tables/table__trial_metrics_agg.parquet`
- `tables/table__length_tradeoff_agg.parquet`
- `plots/plot__sequence_length_tradeoff.pdf`

The sequence-length tradeoff plot reports score trend plus diversity trend as length changes.
The x-axis uses the configured trial factor (`sample.sequence_length`) so sweep intent stays stable even when elite postprocess trims uncovered edges.
The plot marks the base-config sequence length with a subtle vertical line and highlighted baseline point.
Use this to choose a default length for future workspaces with similar TF combinations.

### Clean old Study outputs

Dry-run one run:

```bash
uv run cruncher study clean --workspace demo_pairwise --study length_vs_score --id <study_id>
```

Delete it:

```bash
uv run cruncher study clean --workspace demo_pairwise --study length_vs_score --id <study_id> --confirm
```
