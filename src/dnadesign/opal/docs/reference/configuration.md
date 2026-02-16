# OPAL Configuration (v2)

OPAL reads configuration from `configs/campaign.yaml`.

## Key blocks

- `campaign`: `name`, `slug`, `workdir`
- `data`: `location`, `x_column_name`, `y_column_name`, `y_expected_length`
- `transforms_x`: `{ name, params }` (raw X -> model-ready X)
- `transforms_y`: `{ name, params }` (table -> model-ready Y; CSV/Parquet/XLSX)
- `model`: `{ name, params }`
- `objectives`: list of `{ name, params }` (one or more objective plugins)
- `selection`: `{ name, params }`
- `training`: `policy`
- `training.y_ops`: list of `{ name, params }`
- `ingest`: duplicate handling for label CSVs
- `scoring`: batch sizing
- `safety`: preflight/data guards
- `plot_config`: optional path to a separate plots YAML
- `plot_defaults`, `plot_presets`, `plots`: optional plot-only keys when using inline plot config

## Required v2 selection params

`selection.params` is explicit and channel-driven.

- `top_k`: positive integer
- `score_ref`: `<objective_name>/<score_channel_name>`
- `objective_mode`: `maximize|minimize`
- `tie_handling`: `competition_rank|dense_rank|ordinal`
- `uncertainty_ref`: required when `selection.name: expected_improvement`; must reference a standard-deviation channel

No schema defaults are applied for `top_k`, `objective_mode`, or `tie_handling`. Declare all required selection fields explicitly in YAML.

## Defaults

If a block is omitted, OPAL supplies conservative defaults:

- `ingest.duplicate_policy`: `error`
- `scoring.score_batch_size`: `10000`
- `training.policy`: `{}` and `training.y_ops`: `[]`
- `safety`: fail_on_mixed_biotype_or_alphabet=true, require_biotype_and_alphabet_on_init=true,
  conflict_policy_on_duplicate_ids=error, write_back_requires_columns_present=true, accept_x_mismatch=false

Plugin `params` default to `{}`, but plugin names are required.
Unknown plugin names fail during config load and therefore fail fast in CLI/API entrypoints (`init`, `validate`, `run`, etc.).

## Semantic wiring (model → objective → selection)

1. `model` predicts `y_hat` (and, for GP, predictive standard deviation).
2. Each objective emits named score channels (and optional uncertainty channels).
3. `selection.params.score_ref` chooses the score channel used for ranking.
4. `selection.params.uncertainty_ref` (EI only) chooses the uncertainty standard deviation channel.
5. `selection.params.objective_mode` must match the selected score channel mode.

## Minimal baseline example (RF + top_n)

```yaml
campaign:
  name: "My Campaign"
  slug: "my_campaign"
  workdir: "src/dnadesign/opal/campaigns/my_campaign_dir"

data:
  location: { kind: usr, path: src/dnadesign/usr/datasets, dataset: my_dataset }
  x_column_name: "my_x_column"
  y_column_name: "my_y_column"
  y_expected_length: 1

transforms_x: { name: identity, params: {} }
transforms_y: { name: scalar_from_table_v1, params: {} }

model:
  name: random_forest
  params: { n_estimators: 100, random_state: 7 }

objectives:
  - name: scalar_identity_v1
    params: {}

selection:
  name: top_n
  params:
    top_k: 12
    score_ref: "scalar_identity_v1/scalar"
    objective_mode: maximize
    tie_handling: competition_rank

training:
  policy:
    cumulative_training: true
    label_cross_round_deduplication_policy: latest_only
    allow_resuggesting_candidates_until_labeled: true
```

## UQ example (GP + expected_improvement)

```yaml
model:
  name: gaussian_process
  params:
    alpha: 1.0e-6
    normalize_y: true
    n_restarts_optimizer: 2
    kernel:
      name: matern
      length_scale: 0.5
      nu: 1.5
      with_white_noise: true

objectives:
  - name: sfxi_v1
    params:
      setpoint_vector: [0, 0, 0, 1]
      scaling: { min_n: 1 }

selection:
  name: expected_improvement
  params:
    top_k: 12
    score_ref: "sfxi_v1/sfxi"
    uncertainty_ref: "sfxi_v1/sfxi"
    objective_mode: maximize
    tie_handling: competition_rank
    alpha: 1.0
    beta: 1.0
```

## Precedence and wiring

- `campaign.workdir` and `data.location.path` resolve relative to the campaign root (parent of `configs/`), unless absolute.
- CLI flags override YAML for that invocation:
  `run --k` overrides `selection.params.top_k`, `run --score-batch-size` overrides `scoring.score_batch_size`,
  and `ingest-y --transform/--params` overrides `transforms_y`.
- `--round` is the canonical flag; `--labels-as-of` and `--observed-round` are aliases.
- `transforms_y` is ingest-only; training/prediction uses `transforms_x` plus optional `training.y_ops`.
- `state.json` records resolved config per round; ledger sinks are long-term audit.
- `plot_config` paths resolve relative to the `configs/campaign.yaml` that declares them.
- `plot_defaults` / `plot_presets` / `plots` are consumed by `opal plot`; runtime round execution does not read them.
