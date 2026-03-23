## stress_ethanol_cipro_growth

- Last verified: 2026-03-22
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Study execution map: `pipeline.yaml`
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

- Merged anchor dataset: `promoter/stress_ethanol_cipro_anchor_set` or `n/a`
- Construct-expanded context dataset: `promoter/stress_ethanol_cipro_construct_contexts` or `n/a`
- Canonical full-lane feature dataset: `promoter/stress_ethanol_cipro_feature_matrix` or `n/a`
- Cluster results root: `n/a`
- OPAL config: `n/a`

Current design note: the checked-in Infer full-lane configs keep `anchor_only`
and `template_1kb` as separate jobs and dataset planes. The study still tracks
`promoter/stress_ethanol_cipro_feature_matrix` as the planned shared downstream
feature surface, but the pre-infer execution path is explicit rather than
implicit.

### Infer matrix status

- `anchor_only`: `pending`
  - config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml`
  - 7B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_7b_batch_with_notify.yaml`
  - 20B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_20b_batch_with_notify.yaml`
  - dataset: `promoter/stress_ethanol_cipro_anchor_set`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `anchor_plus_template`: `pending`
  - config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml`
  - 7B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_7b_batch_with_notify.yaml`
  - 20B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_20b_batch_with_notify.yaml`
  - dataset: `promoter/stress_ethanol_cipro_construct_contexts`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `full_lane_set`: `pending`
  - config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.yaml`
  - optional Hopper lane: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.full_lane_set.evo2_20b.yaml`
  - datasets: `promoter/stress_ethanol_cipro_anchor_set`, `promoter/stress_ethanol_cipro_construct_contexts`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
  - model lanes: `evo2_7b`, `evo2_20b`
  - batch note: use the lane-specific presets above because ops auto/resume requires one USR destination per runbook.

### Rollback and maintenance

- Infer reset: `uv run infer prune --usr promoter/stress_ethanol_cipro_feature_matrix --usr-root src/dnadesign/usr/datasets`
- Infer namespace archive: `uv run usr maintenance overlay-remove promoter/stress_ethanol_cipro_feature_matrix --namespace infer --mode archive`
- DenseGen overlay compaction: `uv run usr maintenance overlay-compact densegen/study_stress_ethanol_cipro --namespace densegen`

### Batch and notify

- DenseGen batch route: `src/dnadesign/ops/runbooks/presets/densegen_stress_ethanol_cipro_batch_with_notify.yaml`
- Infer Notify root: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/`
- Infer anchor-only 7B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_7b_batch_with_notify.yaml`
- Infer anchor-plus-template 7B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_7b_batch_with_notify.yaml`
- Infer anchor-only 20B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_20b_batch_with_notify.yaml`
- Infer anchor-plus-template 20B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_20b_batch_with_notify.yaml`
- DenseGen Notify profile or config: `src/dnadesign/densegen/workspaces/study_stress_ethanol_cipro/outputs/notify/densegen/profile.json`
- Infer anchor-only Notify profile or config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_only_7b/profile.json`
- Infer anchor-plus-template Notify profile or config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_plus_template_7b/profile.json`
- Notify secret prerequisite: export `NOTIFY_WEBHOOK_FILE=/usr4/dl523/esouth/.config/dnadesign/notify/secrets/study_stress_ethanol_cipro.webhook` or materialize the same `file://` secret ref into the lane profile before submit.
- Infer Notify profile materialization: use the `setup_command` entries in `pipeline.yaml` under `study_pipeline.infer.notify_watch_contract`.
- TLS prerequisite for `notify profile doctor` and live Slack delivery: export `SSL_CERT_FILE` or store the same CA bundle path in the materialized profile.
- DenseGen watch command: `uv run notify usr-events watch --events src/dnadesign/usr/datasets/densegen/study_stress_ethanol_cipro/.events.log --dry-run --no-advance-cursor-on-dry-run`
- Infer anchor-only resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml --json`
- Infer anchor-plus-template resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml --json`
- Infer Slack message semantics: `attach` sends `running` progress with run id, dataset, chunk rows, and workspace rows; `materialize` sends `success` with `rows_written`.

### Next actions

- Continue DenseGen growth in the shared USR root until the anchor dataset reaches at least `100000` rows.
- Initialize and materialize `promoter/stress_ethanol_cipro_anchor_set` as a fresh shared dataset, then merge in `mg1655_promoters` and `densegen/study_stress_ethanol_cipro` without mutating those source datasets.
- Validate and dry-run `src/dnadesign/construct/workspaces/study_stress_ethanol_cipro_pdual10`, then materialize `promoter/stress_ethanol_cipro_construct_contexts` against the shared `plasmids` dataset.
- Dry-run the checked-in Infer configs in `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/` as soon as the merged anchor and Construct context datasets exist.
- Materialize the lane-specific Infer Notify profiles and submit the 7B anchor-only plus anchor-plus-template presets on a GPU-capable node before attempting the Hopper-only 20B lanes.
