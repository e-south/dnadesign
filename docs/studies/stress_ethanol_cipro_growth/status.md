## stress_ethanol_cipro_growth

- Last verified: 2026-04-13
- Owner: Shockwing
- Affiliated dataset registry: `datasets.yaml`
- Study execution map: `pipeline.yaml`
- USR root: `src/dnadesign/usr/datasets`
- DenseGen source row target: at least `100000` anchor rows before the first canonical consolidated feature-matrix realization
- Current infer-bearing shared handoff datasets: `promoter/stress_ethanol_cipro_anchor_set`, `promoter/stress_ethanol_cipro_construct_contexts`
- Canonical consolidated feature dataset: `promoter/stress_ethanol_cipro_feature_matrix` (`planned`; current Infer writes land on the two shared handoff datasets)
- Current consolidated feature-dataset row count: `n/a`
- Preferred infer model family: `evo2_20b`
- Alternate infer model family: `evo2_7b`

### Source datasets

- DenseGen anchor shared dataset: `densegen/study_stress_ethanol_cipro` (upstream source dataset; use `promoter-study-status` for live local rows, target gap, and sync posture)
- Wildtype or manual dataset: `mg1655_promoters` (`4` rows: `spyp`, `sulAp`, `soxSp`, `J23105`)
- Construct template seed dataset: `plasmids` (`1` row)

### Shared infer-bearing handoff datasets

- Anchor-only handoff dataset: `promoter/stress_ethanol_cipro_anchor_set` (shared anchor-only Infer surface; use `promoter-study-status` for live rows and sync audits)
- Construct-expanded handoff dataset: `promoter/stress_ethanol_cipro_construct_contexts` (shared 1 kb template-backed Infer surface; use `promoter-study-status` for live rows and sync audits)
- Semantic completeness note: as of 2026-04-13 the shared handoff datasets carry the repaired `densegen` namespace as overlays, so `densegen__plan` and `densegen__required_regulators` are visible for all DenseGen-derived rows on both handoff planes while the WT/manual controls remain intentionally unmatched.

### Planned consolidated outputs

- Canonical full-lane feature dataset: `promoter/stress_ethanol_cipro_feature_matrix` (`planned`)
- Cluster results root: `n/a`
- OPAL config: `n/a`

Infer runs are split across two dataset planes: `anchor_only` writes to
`promoter/stress_ethanol_cipro_anchor_set`, and `template_1kb` writes to
`promoter/stress_ethanol_cipro_construct_contexts`. The consolidated feature
matrix remains a later handoff step rather than the current write path. The
Construct side still uses the single `forward_anchor_window` workspace
contract.
Read the row counts by plane. DenseGen source rows measure upstream growth.
Handoff rows measure the shared Infer surfaces. A large handoff row count does
not prove that the DenseGen source target has been met. For this study,
that DenseGen target is now historical context rather than the main status
gate: the current record-backed phase is driven by the shared handoff datasets
and the active Infer lane routing.
Blackwell tuning note: current checked-in lane defaults follow the latest
pressure-test operating points on the Blackwell Evo2 environment, re-verified
on 2026-04-07 with three read-only GPU repeats per lane:
`anchor_only_7b=1024`, `anchor_plus_template_7b=128`,
`anchor_only_20b=256`, `anchor_plus_template_20b=48`. The local-only
full-lane configs inherit the tighter templated limit for each model family
(`7b=128`, `20b=48`) because they share one model-level `batch_size` across
both dataset planes.

Execution note: all checked-in Infer configs target `cuda:0`. A login-node
`infer validate config` or `infer run --dry-run` can still pass while local
GPU capacity is absent, so treat those commands as config-validity checks, not
as proof that the current host can execute Infer directly.
Cold-start note: the first live `evo2_20b` run can spend a visible startup
window in `fetch -> weight hydration -> GPU residency -> first attach events`.
During that window, lack of new USR writes alone is not evidence that Infer is
hung. Check GPU memory growth plus `.events.log` movement before intervening.
Environment portability note: model fit and `.venv` portability are separate.
The current working Evo2 environment should be treated as Blackwell-pinned
until a real `infer extract` smoke proves portability on another GPU family.
`evo2_7b` being eligible for smaller GPUs is not enough to assume that this
specific `.venv` will run there.
Current SCC selector note: the visible Blackwell-family batch lane is
`gpu_t=RTXP6000` with `gpu_capability=12.0` and `gpu_memory=96 GiB`. A generic
`gpu_c=9.0` request can land on H200 instead and is no longer the safe default
for the study's 20B batch presets.

### Infer matrix status

- `anchor_only`: `in_progress`
  - preferred 20B config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_20b.yaml`
  - alternate 7B config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml`
  - tuned Blackwell defaults: `20b batch_size=256`, `7b batch_size=1024`
  - preferred 20B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_20b_batch_with_notify.yaml`
  - alternate 7B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_7b_batch_with_notify.yaml`
  - dataset: `promoter/stress_ethanol_cipro_anchor_set`
  - readiness note: `promoter-study-preflight --scope next --json` is green on a GPU host when `NOTIFY_WEBHOOK_FILE` or `NOTIFY_WEBHOOK` plus `SSL_CERT_FILE` are exported and the 20B preset verifies against `gpu_t=RTXP6000`.
  - live run note: canonical 20B write-back has already started on the shared anchor dataset; inspect the target dataset `.events.log` or infer overlay counts for the current checkpoint before resuming or submitting another anchor-only lane.
  - cold-start signals: treat `nvidia-smi` memory growth, first `attach` events with `completed_rows`, watcher cursor movement, and `spool_files=0` as the healthy startup sequence.
  - outputs expected: `log_likelihood__total`, `log_likelihood__mean_per_token`, `output_layer_mean__seq_mean`, `intermediate_embedding__block23_mlp_out__seq_mean`
- `anchor_plus_template`: `ready`
  - preferred 20B config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_20b.yaml`
  - alternate 7B config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml`
  - tuned Blackwell defaults: `20b batch_size=48`, `7b batch_size=128`
  - preferred 20B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_20b_batch_with_notify.yaml`
  - alternate 7B batch route: `src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_7b_batch_with_notify.yaml`
  - dataset: `promoter/stress_ethanol_cipro_construct_contexts`
  - readiness note: preflight is green on a GPU host with notify env exported when the preset verifies against `gpu_t=RTXP6000`; this is the next recommended lane after the active anchor-only collection or via the notify-enabled batch preset.
  - runtime note: use `h_rt=24:00:00` for the zero-start Blackwell batch preset. The last full dataset run reached `53.27%` at the `12h` cap, which projects to about `22.5h` from zero on the current lane.
  - outputs expected: `log_likelihood__total`, `log_likelihood__mean_per_token`, `output_layer_mean__seq_mean`, `output_layer_mean__anchor_mean`, `intermediate_embedding__block23_mlp_out__seq_mean`, `intermediate_embedding__block23_mlp_out__anchor_mean`
- `full_lane_set`: `local-only`
  - config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.yaml`
  - preferred 20B lane config: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.full_lane_set.evo2_20b.yaml`
  - alternate 7B lane: `src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.full_lane_set.evo2_7b.yaml`
  - tuned local-only defaults: `20b batch_size=48`, `7b batch_size=128`
  - datasets: `promoter/stress_ethanol_cipro_anchor_set`, `promoter/stress_ethanol_cipro_construct_contexts`
  - outputs expected: `ll`, `output_layer_mean`, `intermediate_embedding`
  - model lanes: `evo2_7b`, `evo2_20b`
  - batch note: use the lane-specific presets above because ops auto/resume and Notify expect one USR destination and one watcher surface per run.

### Rollback and maintenance

- Infer reset, anchor-only lane: `uv run infer prune --usr promoter/stress_ethanol_cipro_anchor_set --usr-root src/dnadesign/usr/datasets`
- Infer reset, anchor-plus-template lane: `uv run infer prune --usr promoter/stress_ethanol_cipro_construct_contexts --usr-root src/dnadesign/usr/datasets`
- Infer namespace archive, anchor-only lane: `uv run usr maintenance overlay-remove promoter/stress_ethanol_cipro_anchor_set --namespace infer --mode archive`
- Infer namespace archive, anchor-plus-template lane: `uv run usr maintenance overlay-remove promoter/stress_ethanol_cipro_construct_contexts --namespace infer --mode archive`
- DenseGen overlay compaction: `uv run usr maintenance overlay-compact densegen/study_stress_ethanol_cipro --namespace densegen`
- DenseGen overlay repair, anchor handoff: `uv run usr maintenance overlay-project --src densegen/study_stress_ethanol_cipro --dest promoter/stress_ethanol_cipro_anchor_set --namespace densegen --src-join id --dest-join id --allow-missing`
- DenseGen overlay repair, construct handoff: `uv run usr maintenance overlay-project --src densegen/study_stress_ethanol_cipro --dest promoter/stress_ethanol_cipro_construct_contexts --namespace densegen --src-join id --dest-join construct__anchor_id --allow-missing`

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
- Notify readiness note: on a GPU-capable host with `NOTIFY_WEBHOOK_FILE` or `NOTIFY_WEBHOOK` plus `SSL_CERT_FILE` exported, `promoter-study-preflight --scope next --json` is green (`28 ok, 0 attention` as of 2026-03-30).
- Infer Notify profile materialization: use `uv run ops progress show usr.data-plane.promoter-study-preflight --scope full --json` and inspect the failing `notify.profile.*` command checks plus their `surface_id` and `command` fields, or run `notify setup slack --tool infer --config <checked-in-infer-config>` directly. The study pipeline no longer records separate Infer notify profile paths; Infer derives the lane-specific profile path from the single-lane config contract, so `--profile` should not be needed for the checked-in lane configs.
- Infer Notify scope note: use `--config` per lane, not `--workspace study_stress_ethanol_cipro`, because the workspace default `config.yaml` is a multi-destination full-lane config and is intentionally ambiguous for a single USR events stream.
- TLS prerequisite for `notify profile doctor` and live Slack delivery: export `SSL_CERT_FILE` or store the same CA bundle path in the materialized profile.
- Interactive watcher cold-start: when a live profile exists but its cursor file does not, seed the cursor to the current `.events.log` size before `notify usr-events watch --follow` so Slack does not replay historical Infer events from the same dataset.
- DenseGen watch command: `uv run notify usr-events watch --events src/dnadesign/usr/datasets/densegen/study_stress_ethanol_cipro/.events.log --dry-run --no-advance-cursor-on-dry-run`
- Infer anchor-only 20B resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_20b.yaml --json`
- Infer anchor-plus-template 20B resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_20b.yaml --json`
- Infer anchor-only 7B resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml --json`
- Infer anchor-plus-template 7B resolve command: `uv run notify setup resolve-events --tool infer --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml --json`
- Infer Slack message semantics: `attach` sends `running` progress with run id, dataset, chunk rows, and workspace rows; `materialize` sends `success` with `rows_written`.
- Anti-spam note: the infer notify policy is intentionally sparse. A healthy watcher may stay quiet on stdout and Slack between progress thresholds or heartbeat intervals; trust cursor advancement plus `spool_files=0` over an expectation of one message per flush.
- Concurrency note: the clean maximum is one live infer writer per target
  dataset. For this study, that means `max clean concurrency = 2`: one run on
  `promoter/stress_ethanol_cipro_anchor_set` and one run on
  `promoter/stress_ethanol_cipro_construct_contexts`. Same-dataset mixed-model
  runs are mostly data-safe but share one dataset lock and one `.events.log`,
  so they are second-tier operationally.
- Current pragmatic routing note: keep the current working `.venv` pinned to
  Blackwell-family submit routes for both `evo2_20b` and `evo2_7b` until a
  different GPU family is intentionally rebuilt or proven with a real runtime
  smoke. For the checked-in 7B and 20B batch presets on SCC, that now means
  the exact selector `gpu_t=RTXP6000` with `gpu_capability=12.0`.
- Empirical runtime note: the 2026-04-07 read-only Blackwell pressure test
  completed all four checked-in operating points without OOM or auto-derate.
  Mean wall times were about `21.0s` for `anchor_only_7b=1024`, `16.7s` for
  `anchor_plus_template_7b=128`, `44.9s` for `anchor_only_20b=256`, and
  `51.1s` for `anchor_plus_template_20b=48`. Observed peak GPU residency was
  about `15.4 GiB` for the 7B lanes and `44.9 GiB` for the 20B lanes on the
  RTX PRO 6000 Blackwell lane.

### Next actions

- The source assembly and Construct context phases are already materialized locally. Additional DenseGen growth can continue in parallel if desired, but the checked-in study focus is now active Infer collection.
- Resume the lane-specific Blackwell 20B batch presets from the current shared dataset checkpoints, using the exact SCC selector `gpu_t=RTXP6000`; keep one live writer per target dataset.
- Keep the `anchor_plus_template_20b` preset at `h_rt=24:00:00` for zero-start work; `12:00:00` is still acceptable only when a deliberate resume chain is already planned.
- Use the tuned batch defaults for new resumes (`anchor_only_20b=256`, `anchor_plus_template_20b=48`, `anchor_only_7b=1024`, `anchor_plus_template_7b=128`) and keep the full-lane configs for local composition or dry-runs only.
- On any new GPU host, follow the strict cold-start gate in the Infer workspace docs: current snapshot, green preflight, verified Evo2 runtime import/load, then explicit `infer` namespace registration before first real write-back.
