## study_stress_ethanol_cipro

This workspace is the real Infer surface for the `stress_ethanol_cipro_growth`
study.

It keeps the study's context planes explicit:

- `anchor_only` reads `promoter/stress_ethanol_cipro_anchor_set`
- `template_1kb` reads `promoter/stress_ethanol_cipro_construct_contexts`

That split is deliberate. The current Evo2 `feature_bundle` contract keeps
context kind explicit per job, so the study's "full lane set" is represented as
one config with two jobs, not one mixed-context ingest dataset.

Configs:

- `config.anchor_only.evo2_7b.yaml`
- `config.anchor_plus_template.evo2_7b.yaml`
- `config.anchor_only.evo2_20b.yaml`
- `config.anchor_plus_template.evo2_20b.yaml`
- `config.full_lane_set.evo2_7b.yaml`
- `config.full_lane_set.evo2_20b.yaml`
- `config.yaml` points at the default 7B full-lane set

Portable preflight:

```bash
uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml

uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml

uv run infer validate config \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.full_lane_set.evo2_7b.yaml
```

Once the study-owned datasets exist, dry-run the same configs. For resumable
batch work with Notify, use the lane-specific presets rather than the full-lane
configs because ops auto/resume requires one USR destination per runbook:

```bash
uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_only_7b_batch_with_notify.yaml \
  --repo-root /project/dunlop/esouth/dnadesign

uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_7b_batch_with_notify.yaml \
  --repo-root /project/dunlop/esouth/dnadesign
```

Planning those presets on this node requires the same webhook secret-file
surface DenseGen already uses. Export `NOTIFY_WEBHOOK_FILE` or materialize a
profile with `webhook.source=secret_ref` before submit.

Recommended Infer Notify bootstrap for the real study:

```bash
export NOTIFY_WEBHOOK_FILE=/abs/path/to/study_stress_ethanol_cipro.webhook
export SSL_CERT_FILE=/abs/path/to/ca-bundle.pem

uv run notify setup slack \
  --profile src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_only_7b/profile.json \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify setup slack \
  --profile src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_plus_template_7b/profile.json \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify profile doctor \
  --profile src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_only_7b/profile.json
```

Current Slack delivery semantics come from USR `.events.log` write-back events:

- `attach` emits a `running` update with an Infer-specific message that names
  the run id, dataset id, chunk rows, and workspace row count.
- `materialize` emits a `success` update with the default USR message
  `materialize on <dataset> (rows_written=<n>)`.
- Profile defaults for this study keep `include_args`, `include_context`, and
  `include_raw_event` disabled, so the webhook metadata stays concise unless you
  opt into a richer profile.

Use the matching `anchor_only` and `anchor_plus_template` 20B presets only on
Hopper or H200.
