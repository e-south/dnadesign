# OPAL Configuration

OPAL reads configuration from `configs/campaign.yaml`.

## Key blocks

- `campaign`: `name`, `slug`, `workdir`
- `data`: `location`, `x_column_name`, `y_column_name`, `y_expected_length`
- `transforms_x`: `{ name, params }` (raw X -> model-ready X)
- `transforms_y`: `{ name, params }` (table -> model-ready Y; CSV/Parquet/XLSX)
- `model`: `{ name, params }`
- `objective`: `{ name, params }`
- `selection`: `{ name, params }` (strategy, tie handling, objective mode)
- `training`: `policy`
- `training.y_ops`: list of `{ name, params }` (bespoke transforms applied at train and/or predict time)
- `ingest`: duplicate handling for label CSVs
- `scoring`: batch sizing
- `safety`: preflight/data guards
- `metadata`: optional notes
- `plot_config`: optional path to a separate plots YAML (recommended)

## Defaults

If a block is omitted, OPAL supplies conservative defaults:

- `ingest.duplicate_policy`: `error`
- `scoring.score_batch_size`: `10000`
- `training.policy`: `{}` and `training.y_ops`: `[]`
- `safety`: fail_on_mixed_biotype_or_alphabet=true, require_biotype_and_alphabet_on_init=true,
  conflict_policy_on_duplicate_ids=error, write_back_requires_columns_present=true, accept_x_mismatch=false
- `metadata.notes`: `""`
- `plots`: `[]` (no plotting unless declared)

Plugin `params` default to `{}`, but plugin names are required.

## Minimal example

```yaml
campaign:
  name: "My Campaign"
  slug: "my_campaign"
  workdir: "src/dnadesign/opal/campaigns/my_campaign_dir"

data:
  location: { kind: usr, path: src/dnadesign/usr/datasets, dataset: my_dataset }
  x_column_name: "my_x_column"
  y_column_name: "my_y_column"
  y_expected_length: 4   # enforce Y length on validate/run

ingest:
  duplicate_policy: "error"

transforms_x: { name: my_x_preprocessing_prior_to_model, params: {} }
transforms_y: { name: my_y_preprocessing_prior_to_model, params: {} }

model:
  name: random_forest
  params: { ... }

training:
  policy:
    cumulative_training: true
    label_cross_round_deduplication_policy: "latest_only"
    allow_resuggesting_candidates_until_labeled: true

objective:
  name: my_objective
  params: { ... }

selection:
  name: top_n
  params: { ... }

scoring:
  score_batch_size: 10000

safety:
  fail_on_mixed_biotype_or_alphabet: true
```

## Precedence and wiring

- `campaign.workdir` and `data.location.path` resolve relative to the campaign root
  (parent of `configs/`), unless absolute. Prefer `workdir: "."` for portability.
- CLI flags override YAML for that invocation:
  `run --k` overrides `selection.params.top_k`, `run --score-batch-size` overrides `scoring.score_batch_size`,
  and `ingest-y --transform/--params` (JSON file, `.json`) overrides `transforms_y`.
- `--round` is the canonical flag; `--labels-as-of` and `--observed-round` are aliases.
- `transforms_y` is used for ingest only; model training/prediction uses `transforms_x` plus optional `training.y_ops`.
- `state.json` records the resolved config per round; ledger sinks are the long-term source of truth.
- `plot_config` paths resolve relative to the `configs/campaign.yaml` that declares them.
