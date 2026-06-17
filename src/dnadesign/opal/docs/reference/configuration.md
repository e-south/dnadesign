## OPAL Configuration (v2)

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-17


This page documents the `campaign.yaml` configuration contract used by OPAL runtime and CLI commands.
Use it as the source of truth for required keys, defaults, and model/objective/selection wiring.

### Key blocks

`configs/campaign.yaml` is organized into these top-level blocks:

- `campaign`: `name`, `slug`, `workdir`, optional `description`, optional `metadata`
- `ownership`: optional owner metadata for non-portable study fixtures
- `data`: `location`, `x_column_name`, `y_column_name`, `y_expected_length`,
  optional `candidate_scope`
- `candidate_eligibility`: optional pre-selection rule block for generic
  candidate-level exclusion contracts
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

`campaign.metadata` is metadata only; it does not change runtime behavior.
Generated campaign-set notebooks use scalar metadata values as comparison
facets, so prefer explicit campaign-agnostic keys such as `comparison_group`,
`response_axis`, or `scenario_kind` over study-private scratch labels.

`ownership` is metadata only; it does not change runtime behavior. Use it to
make non-portable checked-in configs explicit, for example:

```yaml
ownership:
  owner_scope: study_fixture
  study_id: stress_ethanol_cipro_growth
  dataset_id: usr_prom_eth_cip_opal_candidates
  portable: false
```

`data.candidate_scope` is an optional scoring-scope contract. Use it when a
campaign should score a declared ID subset while still reading X from the
configured candidate `records.parquet`. The scope file must be a `.parquet`,
`.pq`, or `.csv` table with unique, non-null IDs that all exist in
`records.parquet`.

```yaml
data:
  location: { kind: usr, path: src/dnadesign/usr/datasets, dataset: my_dataset }
  x_column_name: "my_x_column"
  y_column_name: "my_y_column"
  candidate_scope:
    kind: id_list
    path: scopes/heldout_sigma35_ids.parquet
    id_column: id
```

Candidate scopes filter the candidate pool before OPAL excludes already labeled
IDs. They do not change label-source validation, training labels, or X
transforms. This is the preferred way to run split-specific campaigns over a
large shared candidate table without physically copying that table for each
split.

### Candidate eligibility

`candidate_eligibility` is an optional pre-selection contract. OPAL applies
these rules after `candidate_scope` and before excluding already-labeled IDs,
scoring, objective ranking, or selection. Use it for generic, auditable
candidate exclusion rules that should affect every future round. OPAL records
input rows, output rows, filtered counts, rule ids, and parameter hashes in
validation and runtime context reports. A rule that leaves too few candidates
must fail fast rather than silently changing selection behavior.

The first built-in rule is `restriction_site_exclusion`. It scans the assembled
insert defined by a study- or campaign-owned strategy and excludes candidates
with unexpected restriction enzyme motifs. OPAL owns the generic scan/report
contract; the study owns the concrete cloning strategy, flank sequences, and
allowed motif regions.

```yaml
candidate_eligibility:
  rules:
    - name: restriction_site_exclusion
      params:
        sequence_column: sequence
        scan_space: final_assembled_insert
        assembly_strategy_ref: sfxi_promoter_insert:v1
        left_flank: accgggatcctgcag
        right_flank: tgagggaattcgcga
        expected_core_length: 60
        min_remaining_candidates: 1000
        forbidden_sites:
          - enzyme: BamHI
            motif: GGATCC
            allowed_regions: [left_flank]
          - enzyme: EcoRI
            motif: GAATTC
            allowed_regions: [right_flank]
```

For this rule, `sequence_column` must contain the candidate core sequence, not
a context window or final insert. Flanks must be lowercase `acgt`; motifs are
uppercase `ACGT`; `scan_space` currently supports only
`final_assembled_insert`; `on_violation` currently supports only `exclude`.
Allowed regions are exact flank/core regions. A motif spanning a flank/core
junction is not allowed unless a future rule explicitly models that region.

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

Shared sidecar campaigns also fail closed on generic contamination. Non-empty
values in the configured current-Y column, campaign-local observed-label entries
in `opal__<slug>__label_hist`, or any campaign-local label-history entries while
`writeback.prediction_records: ledger_only` are treated as leakage contract
errors. `opal status --json` exposes this as `label_source.leakage` without
loading the configured X column.

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

### Vector target example (RF + one selected channel)

Use this pattern when a study-owned oracle emits a finite numeric target vector,
such as binary TF-family presence, TF-family counts, or one-vs-rest class
columns. OPAL treats the columns as generic numeric target channels; biological
meaning stays in the campaign metadata and source oracle.

```yaml
data:
  y_expected_length: 3

transforms_y:
  name: vector_from_table_v1
  params:
    value_columns: [tf_lexA_presence, tf_cpxR_presence, tf_baeR_presence]

model:
  name: random_forest
  params: { n_estimators: 100, random_state: 7 }

objectives:
  - name: vector_channel_v1
    params:
      channel_index: 0
      channel_name: tf_lexA_presence
      mode: maximize

selection:
  name: top_n
  params:
    top_k: 12
    score_ref: "vector_channel_v1/tf_lexA_presence"
    objective_mode: maximize
    tie_handling: competition_rank
```

### Vector target example (RF + target-vector similarity)

Use this pattern when the objective is closeness to a declared numeric vector
rather than one channel. OPAL still treats the labels as a generic finite
numeric vector; study-specific semantics stay in metadata and docs.

```yaml
data:
  y_expected_length: 4

transforms_y:
  name: vector_from_table_v1
  params:
    value_columns: [v00, v10, v01, v11]

model:
  name: random_forest
  params: { n_estimators: 100, random_state: 7 }

objectives:
  - name: vector_target_similarity_v1
    params:
      target_vector: [0, 0, 1, 1]

selection:
  name: top_n
  params:
    top_k: 12
    score_ref: "vector_target_similarity_v1/negative_mse"
    objective_mode: maximize
    tie_handling: competition_rank
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
