## study_stress_ethanol_cipro

This workspace is the real Infer surface for the `stress_ethanol_cipro_growth`
study.

It keeps the study's context planes explicit:

- `anchor_only` reads `usr_prom_eth_cip_anchor`
- `template_1kb` reads `construct_prom_eth_cip_context`

That split is deliberate. The current Evo2 `feature_bundle` contract keeps
context kind explicit per job, so the study's "full lane set" is represented as
one config with two jobs, not one mixed-context ingest dataset.

Configs:

- `config.anchor_only.evo2_20b.yaml`
- `config.anchor_plus_template.evo2_20b.yaml`
- `config.full_lane_set.evo2_7b.yaml`
- `config.full_lane_set.evo2_20b.yaml`
- `config.sequence_views.main.evo2_7b.yaml`
- `config.sequence_views.reference.evo2_7b.yaml`
- `config.sequence_views.anchor_construct_insert.evo2_7b.yaml`
- `config.sequence_views.context_forward_seq_and_anchor_mean.evo2_7b.yaml`
- `config.sequence_views.context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml`
- `config.sequence_views.reference_analysis_window_core60.evo2_7b.yaml`
- `config.sequence_views.reference_context_forward_seq_and_anchor_mean.evo2_7b.yaml`
- `config.sequence_views.reference_context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml`
- `config.yaml` points at the default 7B full-lane set

Operational unit:

- treat one sequence-view config as one operational unit for real study work:
  one sequence-view dataset, one watcher, one feature-sidecar write surface
- use `config.sequence_views.anchor_construct_insert.evo2_7b.yaml`,
  `config.sequence_views.context_forward_seq_and_anchor_mean.evo2_7b.yaml`, and
  `config.sequence_views.context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml`
  for main-study cold-start, notify, and resumable batch work
- use the three `config.sequence_views.reference_*` configs for reference
  core60 and reference-context Notify lanes; the combined reference config is a
  planning surface because it spans two USR event streams
- keep the multi-job sequence-view config for completion planning; it is not
  the live Notify default because it spans multiple USR event streams
- every 7B sequence-view lane collects intermediate embeddings, mean-pooled
  output-layer logits, and log-likelihoods. Context lanes select both full
  sequence `seq_mean` and bounded `anchor_mean` pooling in the same job so one
  Evo2 forward pass can serve both vector spans. Concat is not an Infer target.

Portable preflight:

```bash
uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.anchor_construct_insert.evo2_7b.yaml

uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.context_forward_seq_and_anchor_mean.evo2_7b.yaml

uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml

uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.reference_analysis_window_core60.evo2_7b.yaml

uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.reference_context_forward_seq_and_anchor_mean.evo2_7b.yaml

uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.reference_context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml

uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.full_lane_set.evo2_7b.yaml

uv run infer validate sequence-view-completion \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.main.evo2_7b.yaml \
  --format json

uv run infer validate sequence-view-completion \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml \
  --format json \
  --max-missing-products 0 \
  --max-stale-vectors 0 \
  --max-stale-scalars 0

uv run notify setup resolve-events \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.anchor_construct_insert.evo2_7b.yaml \
  --json
```

The multi-job sequence-view completion configs are planning surfaces, not live
Notify units. They classify reusable, stale, missing, and product-missing work
for `construct_insert`, forward `realized_context`, reverse-complement
`realized_context`, and reference `analysis_window` views without loading Evo2.
Reusable work is counted only from canonical sequence-view feature/scalar
sidecars. USR row-overlay payload columns are not a coverage source.
`core60_mean`, `seq_mean`, and `anchor_mean` are distinct feature identities.
Exact repeated input sequences still share one Evo2 forward pass through the
`forward_pass_key`; they do not share feature-vector keys unless the full
feature identity is identical.
The lane-specific sequence-view runbooks also render this completion planner as
a pre-submit gate with `--max-missing-products 0 --max-stale-vectors 0
--max-stale-scalars 0`. Missing feature vectors and log-likelihood scalars are
allowed there because they are the work the batch is meant to compute; missing
sequence products, stale vectors, or stale scalar sidecars are not allowed to
slip through as a submit-ready plan.
The local generated `_views/sequence_views.parquet` sidecars now use the generic
product-kind vocabulary. Completion checks will still report attention until
the missing feature vectors and log-likelihood scalars are generated.

Once the study-owned datasets exist, dry-run the same configs. For resumable
batch work with Notify, use the sequence-view presets rather than the multi-job
planning config because Notify requires one USR event stream per runbook:

```bash
uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_anchor_construct_insert_7b_batch_with_notify.yaml \
  --repo-root <repo-root>

uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_context_forward_seq_and_anchor_mean_7b_batch_with_notify.yaml \
  --repo-root <repo-root>

uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_context_reverse_complement_seq_and_anchor_mean_7b_batch_with_notify.yaml \
  --repo-root <repo-root>

uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_reference_analysis_window_core60_7b_batch_with_notify.yaml \
  --repo-root <repo-root>

uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_reference_context_forward_seq_and_anchor_mean_7b_batch_with_notify.yaml \
  --repo-root <repo-root>

uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_sequence_views_reference_context_reverse_complement_seq_and_anchor_mean_7b_batch_with_notify.yaml \
  --repo-root <repo-root>
```

Planning those presets on this node requires the same webhook secret-file
surface DenseGen already uses. Export `NOTIFY_WEBHOOK_FILE` or materialize a
profile with `webhook.source=secret_ref` before submit.

Cold-start gate for first real write-back:

1. current study snapshot:
   `uv run ops progress show usr.data-plane.promoter-study-status --json`
2. current host readiness:
   `NOTIFY_WEBHOOK_FILE=<...> SSL_CERT_FILE=<...> uv run ops progress show usr.data-plane.promoter-study-preflight --scope next --json`
3. live GPU runtime:
   `uv run infer validate config --config <lane-config>`
   `uv run infer run --config <lane-config> --dry-run`
   `uv run infer extract --model-id evo2_20b --device cuda:0 --precision bf16 --alphabet dna --batch-size 1 --fn evo2.log_likelihood --format float --seq ACGTACGTACGT --no-progress`
4. canonical USR namespace registration before first write-back:
   `uv run infer validate usr-registry --config <lane-config>`
   `uv run usr --root src/dnadesign/usr/datasets namespace show infer`
   if `namespace show` fails, run the register command emitted by
   `infer validate usr-registry`

For this workspace, use `--config` per lane for Notify setup/watch. Do not use
`--workspace study_stress_ethanol_cipro` for Notify because the workspace
default `config.yaml` points at a multi-destination full-lane config, which is
intentionally ambiguous for a single USR event stream.

Recommended Infer Notify bootstrap for the real study:

```bash
export NOTIFY_WEBHOOK_FILE=/abs/path/to/study_stress_ethanol_cipro.webhook
export SSL_CERT_FILE=/abs/path/to/ca-bundle.pem

uv run notify setup slack \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.anchor_construct_insert.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify setup slack \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.context_forward_seq_and_anchor_mean.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify setup slack \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify setup slack \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.reference_analysis_window_core60.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify setup slack \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.reference_context_forward_seq_and_anchor_mean.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify setup slack \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.sequence_views.reference_context_reverse_complement_seq_and_anchor_mean.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify profile doctor \
  --profile src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/sequence_views_anchor_construct_insert_7b/profile.json
```

Interactive watcher cold-start for an existing study stream:

```bash
PROFILE=src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/sequence_views_anchor_construct_insert_7b/profile.json
EVENTS=src/dnadesign/usr/datasets/usr_prom_eth_cip_anchor/.events.log
CURSOR=src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/sequence_views_anchor_construct_insert_7b/cursor

mkdir -p "$(dirname "$CURSOR")"
stat -c %s "$EVENTS" > "$CURSOR"
uv run notify usr-events watch --profile "$PROFILE" --follow --idle-timeout 7200
```

Seed the cursor only when you want a new watcher to start at the current end
of an existing event stream. If you want replay, do not preseed it.

Current Slack delivery semantics come from USR `.events.log` write-back events:

- `attach` emits a `running` update with an Infer-specific message that names
  the run id, dataset id, chunk rows, and workspace row count.
- `materialize` emits a `success` update with the default USR message
  `materialize on <dataset> (rows_written=<n>)`.
- Profile defaults for this study keep `include_args`, `include_context`, and
  `include_raw_event` disabled, so the webhook metadata stays concise unless you
  opt into a richer profile.
- Running updates are intentionally sparse. Do not expect one Slack message or
  one watcher stdout line per attach or per flush. A healthy watcher can stay
  quiet while the cursor advances and the spool stays empty.

Hydration versus hung for `evo2_20b`:

- expected startup path is `fetch -> weight hydration -> GPU residency ->
  first attach events`
- before declaring a hang, check:
  - `nvidia-smi` shows the infer process and rising/stable memory residency
  - the target dataset `.events.log` gains new `attach` events
  - the watcher cursor advances
  - the watcher spool remains empty

Resume semantics:

- `overwrite: false` prevents recomputing already-written feature outputs
- reruns may still backfill metadata fields when stored values are null
- treat "features skipped, metadata filled" as healthy resume behavior rather
  than feature duplication

Use the matching `anchor_only` and `anchor_plus_template` 20B presets only on
GPU lanes that satisfy the checked-in 20B contract. For the current
Blackwell-pinned study environment on SCC, that means the exact selector
`gpu_t=RTXP6000` with `gpu_capability=12.0`; a generic `gpu_capability >= 9.0`
request can still land on H200 instead.

Current Blackwell operating points:

- verified on 2026-04-07 with three read-only repeats per lane on the local
  RTX PRO 6000 Blackwell GPU
- `anchor_only_7b`: `batch_size=1024`, mean wall time about `21.0s`,
  observed peak residency about `15.4 GiB`
- `anchor_plus_template_7b`: `batch_size=128`, mean wall time about `16.7s`,
  observed peak residency about `15.4 GiB`
- `anchor_only_20b`: `batch_size=256`, mean wall time about `44.9s`,
  observed peak residency about `44.9 GiB`
- `anchor_plus_template_20b`: `batch_size=48`, mean wall time about `51.1s`,
  observed peak residency about `44.9 GiB`
- none of those repeats hit `RuntimeOOMError` or auto-derated

Walltime note:

- the `anchor_plus_template_20b` zero-start SCC preset should use
  `h_rt=24:00:00`
- the last full-dataset `12h` run reached only `53.27%`, which projects to
  about `22.5h` from zero on the current lane
