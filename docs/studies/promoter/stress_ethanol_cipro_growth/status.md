## stress_ethanol_cipro_growth

- Last verified: 2026-03-21
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- USR root: `src/dnadesign/usr/datasets`
- Target row count: at least `100000` DenseGen anchor rows before the first shared feature-matrix realization
- Current shared feature dataset: `n/a`
- Current feature-dataset row count: `n/a`

### Source datasets

- DenseGen anchor shared dataset: `densegen/study_stress_ethanol_cipro` (`77680` rows, written directly to the shared USR root)
- Wildtype or manual dataset: `mg1655_promoters` (`4` rows)
- Construct template seed dataset: `plasmids` (`1` row)
- Optional construct context dataset: `n/a`

### Shared downstream datasets

- Anchor-only feature dataset: `promoter/stress_ethanol_cipro_feature_matrix` or `n/a`
- Construct-expanded feature dataset: `promoter/stress_ethanol_cipro_construct_feature_matrix` or `n/a`
- Cluster results root: `n/a`
- OPAL config: `n/a`

### Infer matrix status

- `anchor_only`: `pending`
  - config: `n/a`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `anchor_plus_template`: `pending`
  - config: `n/a`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `full_lane_set`: `pending`
  - config: `n/a`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
  - model lanes: `evo2_7b`, `evo2_20b`

### Rollback and maintenance

- Infer reset: `uv run infer prune --usr promoter/stress_ethanol_cipro_feature_matrix --usr-root src/dnadesign/usr/datasets`
- Infer namespace archive: `uv run usr maintenance overlay-remove promoter/stress_ethanol_cipro_feature_matrix --namespace infer --mode archive`
- DenseGen overlay compaction: `uv run usr maintenance overlay-compact densegen/study_stress_ethanol_cipro --namespace densegen`

### Batch and notify

- DenseGen batch route: `src/dnadesign/ops/runbooks/presets/densegen_stress_ethanol_cipro_batch_with_notify.yaml`
- Infer batch route: `n/a`
- Notify profile or config: `n/a`
- Watch command: `uv run notify usr-events watch --events src/dnadesign/usr/datasets/densegen/study_stress_ethanol_cipro/.events.log --dry-run --no-advance-cursor-on-dry-run`

### Next actions

- Continue DenseGen growth in the shared USR root until the anchor dataset reaches at least `100000` rows.
- Materialize the first shared feature dataset by assembling `densegen/study_stress_ethanol_cipro`, `mg1655_promoters`, and any required construct-context outputs.
- Author and dry-run infer configs for `anchor_only` and `anchor_plus_template` before the first write-back into the shared feature dataset.
