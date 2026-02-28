## Study: Diversity vs Score (Two-TF Demo)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23


### Contents
- [What this Study measures](#what-this-study-measures)
- [Run it](#run-it)
- [Read the outputs](#read-the-outputs)
- [Clean old Study outputs](#clean-old-study-outputs)

This guide uses the `demo_pairwise` workspace Study spec:

- `workspaces/demo_pairwise/configs/studies/diversity_vs_score.study.yaml`

### What this Study measures

This Study isolates selection-policy behavior by replaying MMR diversity settings after sampling.
It answers: how does the score/diversity tradeoff move as `sample.elites.select.diversity` changes?

The sweep is replay-first:

- sampling runs once per replicate seed
- MMR replay sweeps diversity values without re-sampling
- replay uses a dense diversity grid (`0.00` to `1.00` in `0.05` steps)
- replay diversity values must include the base-config diversity value from `configs/config.yaml`

This keeps runtime and artifact bloat lower than brute-force re-sampling each diversity setting.

### Run it

Prerequisite:

- the workspace must already have lock/parse-ready motif data (for `demo_pairwise`, run the two-TF demo data-prep sequence first, including merged motif discovery into `demo_merged_meme_oops`).

```bash
set -euo pipefail
cd src/dnadesign/cruncher/workspaces/demo_pairwise

uv run cruncher lock configs/config.yaml
uv run cruncher parse configs/config.yaml
uv run cruncher study run --spec configs/studies/diversity_vs_score.study.yaml
uv run cruncher study list
```

Find the generated run directory:

```bash
uv run cruncher study show --run outputs/studies/diversity_vs_score/<study_id>
open outputs/plots/study__diversity_vs_score__<study_id>__plot__mmr_diversity_tradeoff.pdf
```

### Read the outputs

Primary artifacts:

- `tables/table__trial_metrics.parquet`
- `tables/table__trial_metrics_agg.parquet`
- `tables/table__mmr_tradeoff_agg.parquet`
- `plots/plot__mmr_diversity_tradeoff.pdf`

The diversity tradeoff plot shows how selected-sequence score changes against diversity metrics over replayed MMR settings.
The plot marks the base-config diversity value with a subtle vertical line and highlighted baseline point.
Use this to select a practical default diversity value for similar studies/workspaces.

### Clean old Study outputs

Dry-run all runs for this Study:

```bash
uv run cruncher study clean --workspace demo_pairwise --study diversity_vs_score --all
```

Delete all runs:

```bash
uv run cruncher study clean --workspace demo_pairwise --study diversity_vs_score --all --confirm
```
