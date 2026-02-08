# Notifications

`dnadesign` includes a tool-agnostic notifier CLI for sending webhook updates from batch jobs or local runs.

Operator onboarding for USR event watchers lives in [Notify USR events operator manual](notify/usr_events.md).

## CLI

```
notify send \
  --provider slack \
  --status success \
  --tool densegen \
  --run-id demo_meme_three_tfs \
  --url-env DENSEGEN_WEBHOOK_URL \
  --message "Run complete"
```

Supported providers:
- `generic` (JSON payload)
- `slack` (text payload)
- `discord` (text payload)

Exactly one of `--url`, `--url-env`, or `--secret-ref` is required. The notifier fails fast on missing inputs.

## Metadata

Include additional metadata with a JSON file:

```
notify send \
  --provider generic \
  --status failure \
  --tool infer \
  --run-id evo2_001 \
  --url-env INFER_WEBHOOK_URL \
  --meta outputs/meta/run_manifest.json
```

The notifier expects a JSON object in the file and attaches it to the `meta` field.

## Usage Patterns

The notifier is intentionally tool-agnostic. Wire it up in scripts, notebooks, or batch jobs
without adding tool-specific flags.

Example: wrap a pipeline step and notify on success/failure.

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="/path/to/config.yaml"

if uv run dense run -c "$CONFIG" --fresh; then
  notify send \
    --provider slack \
    --status success \
    --tool densegen \
    --run-id demo_meme_three_tfs \
    --url-env DENSEGEN_WEBHOOK_URL \
    --message "DenseGen run completed"
else
  notify send \
    --provider slack \
    --status failure \
    --tool densegen \
    --run-id demo_meme_three_tfs \
    --url-env DENSEGEN_WEBHOOK_URL \
    --message "DenseGen run failed"
  exit 1
fi
```

Example: milestone notifications.

```bash
notify send --provider generic --status started --tool densegen --run-id demo --url-env DENSEGEN_WEBHOOK_URL \
  --message "Stage-A build starting"
uv run dense stage-a build-pool -c "$CONFIG"
notify send --provider generic --status running --tool densegen --run-id demo --url-env DENSEGEN_WEBHOOK_URL \
  --message "Stage-B solve starting"
uv run dense run -c "$CONFIG" --resume
notify send --provider generic --status success --tool densegen --run-id demo --url-env DENSEGEN_WEBHOOK_URL \
  --message "Run finished"
```

## Operator Runbooks (Canonical)

To avoid setup drift, USR-event onboarding lives in one canonical runbook:

- Canonical runbook: [Notify USR events operator manual](notify/usr_events.md)
- DenseGen local end-to-end demo: [DenseGen -> USR -> Notify demo](../src/dnadesign/densegen/docs/demo/demo_usr_notify.md)
- BU SCC batch + Notify runbook: [BU SCC Batch + Notify runbook](hpc/bu_scc_batch_notify.md)

Use that runbook for:
- wizard-first setup (`profile wizard`)
- secure secret wiring (`--secret-source auto|env`)
- two-terminal watch workflow (`doctor` -> `dry-run` -> `follow`)
- strict USR `.events.log` path contract
- spool/drain recovery patterns

## Dry Run

To preview the payload:

```
notify send \
  --provider slack \
  --status running \
  --tool densegen \
  --run-id demo \
  --url https://example.com/webhook \
  --dry-run
```
