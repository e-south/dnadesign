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

- `config.anchor_only.evo2_7b.yaml`
- `config.anchor_plus_template.evo2_7b.yaml`
- `config.anchor_only.evo2_20b.yaml`
- `config.anchor_plus_template.evo2_20b.yaml`
- `config.full_lane_set.evo2_7b.yaml`
- `config.full_lane_set.evo2_20b.yaml`
- `config.yaml` points at the default 7B full-lane set

Operational unit:

- treat one lane config as one operational unit for real study work:
  one dataset, one watcher, one resume surface
- use `anchor_only` or `anchor_plus_template` for cold-start, notify, and
  resume work
- keep the full-lane configs for local composition, validation, or dry-runs;
  they are the wrong default for live notify or resumable study execution
  because they span multiple USR destinations

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
  --repo-root <repo-root>

uv run ops runbook plan \
  --runbook src/dnadesign/ops/runbooks/presets/infer_stress_ethanol_cipro_anchor_plus_template_7b_batch_with_notify.yaml \
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
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_only.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify setup slack \
  --tool infer \
  --config src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/config.anchor_plus_template.evo2_7b.yaml \
  --secret-source file \
  --secret-ref "file://$NOTIFY_WEBHOOK_FILE"

uv run notify profile doctor \
  --profile src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_only_7b/profile.json
```

Interactive watcher cold-start for an existing study stream:

```bash
PROFILE=src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_only_20b/profile.json
EVENTS=src/dnadesign/usr/datasets/usr_prom_eth_cip_anchor/.events.log
CURSOR=src/dnadesign/infer/workspaces/study_stress_ethanol_cipro/outputs/notify/infer/anchor_only_20b/cursor

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
