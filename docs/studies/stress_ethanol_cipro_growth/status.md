## stress_ethanol_cipro_growth

- Last verified: 2026-03-24
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Study execution map: `pipeline.yaml`
- USR root: `src/dnadesign/usr/datasets`
- Target row count: at least `100000` DenseGen anchor rows before the first shared feature-matrix realization
- Current shared feature dataset: `n/a`
- Current feature-dataset row count: `n/a`
- Preferred infer model family: `evo2_20b`
- Alternate infer model family: `evo2_7b`

### Source datasets

- DenseGen anchor shared dataset: `densegen/study_stress_ethanol_cipro` (`130864` rows, written directly to the shared USR root)
- Wildtype or manual dataset: `mg1655_promoters` (`4` rows: `spyP_MG1655`, `sulAp`, `soxS`, `J23105`)
- Construct template seed dataset: `plasmids` (`1` row)
- Shared merged anchor dataset: `promoter/stress_ethanol_cipro_anchor_set` (`130868` rows)
- Shared Construct context dataset: `promoter/stress_ethanol_cipro_construct_contexts` (`130868` rows)

### Shared downstream datasets

- Merged anchor dataset: `promoter/stress_ethanol_cipro_anchor_set` (`130868` rows)
- Construct-expanded context dataset: `promoter/stress_ethanol_cipro_construct_contexts` (`130868` rows, 1 kb realized outputs)
- Canonical full-lane feature dataset: `promoter/stress_ethanol_cipro_feature_matrix` or `n/a`
- Cluster results root: `n/a`
- OPAL config: `n/a`

Current design note: the checked-in Infer full-lane configs keep `anchor_only`
and `template_1kb` as separate jobs and dataset planes. The study still tracks
`promoter/stress_ethanol_cipro_feature_matrix` as the planned shared downstream
feature surface, but the pre-infer execution path is explicit rather than
implicit. The study-owned Construct surface is one
`forward_anchor_window` workspace project; the placement contract lives in the
checked-in Construct config and workspace registry rather than being duplicated
throughout the study note.

Execution note: all checked-in Infer configs target `cuda:0`. A login-node
`infer validate config` or `infer run --dry-run` can still pass while local
GPU capacity is absent, so treat those commands as config-validity checks, not
as proof that the current host can execute Infer directly.

### Infer matrix status

- `anchor_only`: `batch-prep`
  - preferred 20B config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_20b.yaml`
  - alternate 7B config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml`
  - preferred 20B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_20b_batch_with_notify.yaml`
  - alternate 7B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_7b_batch_with_notify.yaml`
  - dataset: `promoter/stress_ethanol_cipro_anchor_set`
  - readiness note: config validation is green; notify profiles plus webhook/TLS env are still missing; direct execution still requires a GPU host
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `anchor_plus_template`: `batch-prep`
  - preferred 20B config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_20b.yaml`
  - alternate 7B config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml`
  - preferred 20B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_20b_batch_with_notify.yaml`
  - alternate 7B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_7b_batch_with_notify.yaml`
  - dataset: `promoter/stress_ethanol_cipro_construct_contexts`
  - readiness note: config validation is green; notify profiles plus webhook/TLS env are still missing; direct execution still requires a GPU host
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
- `full_lane_set`: `pending`
  - config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.yaml`
  - preferred Hopper lane: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.full_lane_set.evo2_20b.yaml`
  - alternate 7B lane: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.full_lane_set.evo2_7b.yaml`
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
- Infer anchor-only 20B derived Notify profile path: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_only_20b/profile.json`
- Infer anchor-plus-template 20B derived Notify profile path: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_plus_template_20b/profile.json`
- Infer anchor-only 7B derived Notify profile path: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_only_7b/profile.json`
- Infer anchor-plus-template 7B derived Notify profile path: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_plus_template_7b/profile.json`
- Notify secret prerequisite: export `NOTIFY_WEBHOOK_FILE=/usr4/dl523/esouth/.config/dnadesign/notify/secrets/study_stress_ethanol_cipro.webhook` or materialize the same `file://` secret ref into the lane profile before submit.
- Infer Notify profile materialization: use `uv run ops progress show usr.data-plane.promoter-study-preflight --scope full --json` and inspect the failing `notify.profile.*` command checks plus their `surface_id` and `command` fields, or run `notify setup slack --tool infer --config <checked-in-infer-config>` directly. The study pipeline no longer records separate Infer notify profile paths; Infer derives the lane-specific profile path from the single-lane config contract, so `--profile` should not be needed for the checked-in lane configs.
- Infer Notify scope note: use `--config` per lane, not `--workspace study_stress_ethanol_cipro`, because the workspace default `config.yaml` is a multi-destination full-lane config and is intentionally ambiguous for a single USR events stream.
- TLS prerequisite for `notify profile doctor` and live Slack delivery: export `SSL_CERT_FILE` or store the same CA bundle path in the materialized profile.
- DenseGen watch command: `uv run notify usr-events watch --events src/dnadesign/usr/datasets/densegen/study_stress_ethanol_cipro/.events.log --dry-run --no-advance-cursor-on-dry-run`
- Infer anchor-only 20B resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_20b.yaml --json`
- Infer anchor-plus-template 20B resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_20b.yaml --json`
- Infer anchor-only 7B resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml --json`
- Infer anchor-plus-template 7B resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml --json`
- Infer Slack message semantics: `attach` sends `running` progress with run id, dataset, chunk rows, and workspace rows; `materialize` sends `success` with `rows_written`.

### Next actions

- The source assembly and Construct context phases are already materialized locally. Additional DenseGen growth can continue in parallel if desired, but the checked-in study focus is now Infer batch preparation.
- Materialize the lane-specific Infer Notify profiles for both 20B and 7B configs. Treat 20B as the preferred default and 7B as the alternate path.
- Export `NOTIFY_WEBHOOK_FILE` and `SSL_CERT_FILE` before rerunning `notify profile doctor` or `ops runbook plan`.
- Use `uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json` to focus on the infer-batch-preparation blockers rather than the full historical study surface.
- Submit the preferred 20B anchor-only and anchor-plus-template presets on Hopper/H200-capable GPU infrastructure when available; use the 7B presets on other GPU-capable nodes when a non-Hopper lane is needed.
