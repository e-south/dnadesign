## Studies guide

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Why studies exist](#why-studies-exist)
- [Spec layout](#spec-layout)
- [Choosing sweep styles](#choosing-sweep-styles)
- [Artifact strategy](#artifact-strategy)
- [Outputs](#outputs)
- [Standard command sequence](#standard-command-sequence)

### Why studies exist

`configs/config.yaml` remains deterministic for normal workspace runs. Study intent (sweep factors, replicate seeds, replay knobs) lives in a separate `*.study.yaml` file so diagnostics/tuning sweeps do not clutter the main config.

Use `portfolio` (see `guides/portfolio_aggregation.md`) when you need cross-workspace aggregation for experimental handoff. `study` remains workspace-local by contract.

Use studies when you need:

- reproducible parameter sweeps (`trials x seeds`)
- optional grid expansion (`trial_grids`) when explicit trial lists are verbose
- higher-order tradeoff plots (for example diversity vs score, sequence length vs score/diversity)
- aggregate outputs that combine multiple runs into one reportable artifact set

### Spec layout

Store study specs inside the workspace:

```text
<workspace>/configs/studies/<name>.study.yaml
```

Minimal shape:

```yaml
study:
  schema_version: 3
  name: diversity_vs_score
  base_config: config.yaml
  target:
    kind: regulator_set
    set_index: 1
  execution:
    parallelism: 6
    on_trial_error: continue
    exit_code_policy: nonzero_if_any_error
    summarize_after_run: true
  artifacts:
    trial_output_profile: analysis_ready
  replicates:
    seed_path: sample.seed
    seeds: [1, 2, 3]
  trials:
    - id: L16
      factors:
        sample.sequence_length: 16
    - id: L20
      factors:
        sample.sequence_length: 20
  trial_grids:
    - id_prefix: G
      factors:
        sample.sequence_length: [16, 20]
        sample.elites.select.diversity: [0.0, 0.5]
  replays:
    mmr_sweep:
      enabled: true
      pool_size_values: [auto]
      diversity_values: [0.0, 0.25, 0.5, 0.75, 1.0]
```

Strict contracts:

- unknown keys fail (`extra=forbid`)
- trial IDs must be unique and slug-safe
- `trial_grids[].id_prefix` values must be unique and slug-safe
- `trial_grids[].factors` must be non-empty dot-path -> non-empty value-list mappings
- study factors are strict and sweep-only; only these keys are allowed:
  - `sample.sequence_length`
  - `sample.elites.select.diversity`
  - `sample.optimizer.chains`
  - `sample.optimizer.cooling.stages`
  - `sample.moves.overrides.proposal_adapt.freeze_after_beta`
  - `sample.moves.overrides.gibbs_inertia.p_stay_end`
  - `sample.moves.overrides.move_schedule.end`
- `base_config` must exist (no CWD fallback)
- `execution.parallelism` must be an integer `>= 1`
- when omitted, `execution.parallelism` defaults to `6`
- at least one trial source is required (`trials` or `trial_grids`)
- trial expansion is bounded (`<=500` combinations per grid and `<=500` total expanded trials)
- every swept factor must include the base-config value somewhere in the study domain
- when replay is enabled, `replays.mmr_sweep.diversity_values` must include the base-config diversity value

### Choosing sweep styles

#### Sequence-length sweeps (sampling sweep)

Change `sample.sequence_length` in `trials[].factors` or `trial_grids[].factors`. This re-runs sampling because the state dimension changes.

#### Diversity sweeps (replay sweep)

Use `replays.mmr_sweep` to replay MMR selection over diversity/pool grids from saved sample artifacts. This avoids resampling for knobs that only affect selection policy.

#### Grid expansion

Use `trial_grids` when you want cartesian expansion from compact axis definitions. IDs are generated as `<id_prefix>_<n>` in deterministic order.

Important contract:

- `trials[].factors` are not inherited by `trial_grids`.
- studies inherit non-swept behavior from `configs/config.yaml`; do not duplicate non-sweep knobs in study factors.
- workspace `length_vs_score` studies use step-2 sequence-length axes and include the base `sample.sequence_length` as an anchor value.

### Artifact strategy

Study output root is deterministic:

```text
<workspace>/outputs/studies/<study_name>/<study_id>/
```

`study_id` is derived from frozen spec + base config hash + target descriptor (no timestamp identity).

Inside a study run:

- `study/` frozen spec, study manifest/status, logs
- `trials/` per trial/seed sample outputs
- `tables/` aggregate parquet tables
- `outputs/plots/study__<study_name>__<study_id>__plot__*` namespaced aggregate plots
- `manifests/` table/plot manifests

Study trial runs are intentionally excluded from workspace `run_index.json`.

Study preflight validates lockfile + target readiness and parse-readiness (motif loading for each targeted set) before any trial execution.
When `execution.parallelism > 1`, trial sampling is fanned out in bounded worker processes while study manifest/status writes remain single-writer in the coordinator.

Profile notes:

- `minimal` disables heavy artifacts such as trace capture but still keeps required sequence tables.
- `analysis_ready` preserves broader artifacts for downstream `cruncher analyze`.
- MMR replay (`replays.mmr_sweep.enabled=true`) requires `sample.output.save_sequences=true` for every trial after factors/profile application.

### Outputs

Core aggregate outputs:

- `tables/table__trial_metrics.parquet`
- `tables/table__trial_metrics_agg.parquet`
- `tables/table__length_tradeoff_agg.parquet`
- `tables/table__mmr_tradeoff_agg.parquet` (when replay enabled)
- `outputs/plots/study__<study_name>__<study_id>__plot__sequence_length_tradeoff.pdf` (when sequence length varies across successful trials)
- `outputs/plots/study__<study_name>__<study_id>__plot__mmr_diversity_tradeoff.pdf` (when replay enabled)

Both tradeoff plots annotate the base-config x-value with:
- a subtle vertical reference line
- a highlighted baseline point (same hue as the series, black edge)

When `study summarize --allow-partial` is used, aggregate tables include `n_missing_*` columns and status warnings record the missing-data breakdown (`non_success`, `missing_run_dirs`, `missing_metric_artifacts`, `missing_mmr_tables`).
If the frozen Study spec uses `execution.exit_code_policy: nonzero_if_any_error`, partial summarize exits non-zero after writing outputs.

### Standard command sequence

```bash
set -euo pipefail
cd <workspace>
cruncher study list
cruncher study run --spec configs/studies/length_vs_score.study.yaml --force-overwrite
cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite
cruncher study show --run outputs/studies/length_vs_score/<study_id>
cruncher study show --run outputs/studies/diversity_vs_score/<study_id>
open outputs/plots/study__length_vs_score__<study_id>__plot__sequence_length_tradeoff.pdf
open outputs/plots/study__diversity_vs_score__<study_id>__plot__mmr_diversity_tradeoff.pdf
```

`study list` and `workspaces list` treat studies as workspace-scoped entities:

- Study intent files live in `<workspace>/configs/studies/`.
- Study outputs live in `<workspace>/outputs/studies/`.
- Studies are not separate top-level workspace roots.

Two standard two-TF examples:

- [Study: length vs score](study_length_vs_score.md)
- [Study: diversity vs score](study_diversity_vs_score.md)
