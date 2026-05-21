## OPAL Configuration (v2)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-05-17


This page documents the `campaign.yaml` configuration contract used by OPAL runtime and CLI commands.
Use it as the source of truth for required keys, defaults, and model/objective/selection wiring.

### Key blocks

`configs/campaign.yaml` is organized into these top-level blocks:

- `campaign`: `name`, `slug`, `workdir`
- `data`: `location`, `x_column_name`, `y_column_name`, `y_expected_length`
- `labels`: optional training-label source contract; defaults to
  campaign-scoped label history
- `transforms_x`: `{ name, params }` (raw X -> model-ready X)
- `transforms_y`: `{ name, params }` (table -> model-ready Y; CSV/Parquet/XLSX)
- `model`: `{ name, params }`
- `objectives`: list of `{ name, params }` (one or more objective plugins)
- `selection`: `{ name, params }`
- `training`: `policy`
- `training.y_ops`: list of `{ name, params }`
- `ingest`: duplicate handling for label CSVs
- `scoring`: batch sizing
- `writeback`: prediction writeback policy
- `safety`: preflight/data guards
- `plot_config`: optional path to a separate plots YAML
- `plot_defaults`, `plot_presets`, `plots`: optional plot-only keys when using inline plot config

`plot_config` configures review and inspection artifacts. Runtime round
execution does not depend on plot rendering; failed plots should fail the plot
command and surface in manifests, not silently change model training,
prediction, objective scoring, or selection.

### Required v2 selection params

`selection.params` is explicit and channel-driven.

- `top_k`: positive integer
- `score_ref`: `<objective_name>/<score_channel_name>`
- `objective_mode`: `maximize|minimize`
- `tie_handling`: `competition_rank|dense_rank|ordinal`
- `uncertainty_ref`: required when `selection.name: expected_improvement`; must reference a standard-deviation channel

Built-in schemas currently provide defaults for `top_k`, `objective_mode`, and `tie_handling`. For deterministic behavior and clearer reviews, declare all four keys explicitly in YAML.

### Defaults

If an optional block is omitted, OPAL supplies conservative defaults:

- `labels.source.kind`: `campaign_history`
- `labels.id_column`: `id`; `labels.round_column`: `observed_round`;
  `labels.batch_column`: `batch_id`; `labels.dedup_policy`: `latest_by_round`
- `ingest.duplicate_policy`: `error`
- `scoring.score_batch_size`: `10000`
- `writeback.prediction_records`: `label_history` for `campaign_history`
  campaigns; shared `usr_sidecar` campaigns must declare this explicitly
- `training.policy`: `{}` and `training.y_ops`: `[]`
- `safety`: fail_on_mixed_biotype_or_alphabet=true, require_biotype_and_alphabet_on_init=true,
  conflict_policy_on_duplicate_ids=error, write_back_requires_columns_present=true,
  accept_x_mismatch=false, max_x_matrix_gib=8.0

`safety.max_x_matrix_gib` is a fail-fast memory budget for model-ready X batches
during `opal run`. For `writeback.prediction_records: ledger_only`, OPAL loads
record metadata without the configured X column, streams candidate X from
Parquet in `scoring.score_batch_size` batches, and aborts if a single train plus
score batch would exceed this budget. For
`writeback.prediction_records: label_history`, OPAL still needs a full records
frame to rewrite prediction history into `records.parquet`; large candidate
universes should use `ledger_only` unless record-level prediction writeback is
explicitly required.

Plugin `params` default to `{}`, but plugin names are required.
Unknown plugin names fail at `opal validate` (registry resolution is strict).
Duplicate YAML keys fail fast during config parsing (for example, two `objectives:` blocks).
Analytical `sfxi_v1` uncertainty requires `logic_exponent_beta=1` and `intensity_exponent_gamma=1`; invalid combinations are rejected at config load.

### Semantic wiring (model → objective → selection)

1. The configured label source supplies training `Y` through the requested
   `--labels-as-of` round.
2. `model` predicts `y_hat` (and, for GP, predictive standard deviation).
3. Each objective emits named score channels (and optional uncertainty channels).
4. `selection.params.score_ref` chooses the score channel used for ranking.
5. `selection.params.uncertainty_ref` (EI only) chooses the uncertainty standard deviation channel.
6. `selection.params.objective_mode` must match the selected score channel mode.

### Shared label source example

Use this when multiple campaigns share candidate identity, `X`, and observed
assay labels, but differ in objective/setpoint/selection settings.

```yaml
labels:
  source:
    kind: usr_sidecar
    dataset: my_dataset
    path: _opal/observed_labels.parquet
  y_space: sfxi_vec8
  id_column: id
  round_column: observed_round
  batch_column: batch_id
  dedup_policy: latest_by_round
writeback:
  prediction_records: ledger_only
```

`usr_sidecar` label sources must point at the same dataset as
`data.location.dataset` and must explicitly declare
`writeback.prediction_records`. Use `ledger_only` when the shared
`records.parquet` should remain a candidate/X table and prediction outputs
should live in campaign ledgers. OPAL fails fast rather than falling back to
campaign-local label history when a configured shared sidecar is missing or
malformed during `run`/`explain`; `opal validate` reports missing pre-ingest
sidecars and validates schema when the file exists.

### Minimal baseline example (RF + top_n)

Use this example when your objective is already scalar and selection is deterministic.

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

### UQ example (GP + expected_improvement)

Use this example when selection must consume both score and uncertainty channels.

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

### Precedence and wiring

Resolution and override rules:

- `campaign.workdir` and `data.location.path` resolve relative to the campaign root (parent of `configs/`), unless absolute.
- CLI flags override YAML for that invocation:
  `run --k` overrides `selection.params.top_k`, `run --score-batch-size` overrides `scoring.score_batch_size`,
  and `ingest-y --transform/--params` overrides `transforms_y`.
- `--round` is a shared alias across commands; prefer explicit flags in scripts:
  `ingest-y --observed-round` for label stamping and `run/explain --labels-as-of` for training cutoff.
- `transforms_y` is ingest-only; training/prediction uses `transforms_x` plus optional `training.y_ops`.
- `state.json` records resolved config per round; ledger sinks are long-term audit.
- `plot_config` paths resolve relative to the `configs/campaign.yaml` that declares them.
- `plot_defaults` / `plot_presets` / `plots` are consumed by `opal plot`; runtime round execution does not read them.
