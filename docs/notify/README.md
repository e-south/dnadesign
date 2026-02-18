# Notify Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-18

`dnadesign` includes a tool-agnostic notifier CLI for webhook delivery from local and batch workflows.

## Choose a workflow

- One-off notifications from scripts, notebooks, or jobs: use `notify send` (this page).
- Long-running USR `.events.log` watcher operations: use [Notify USR events operator manual](usr-events.md).
- BU SCC scheduler deployment patterns for watchers: use [BU SCC Batch + Notify runbook](../bu-scc/batch-notify.md).

## `notify send` quick usage

```bash
notify send \
  --provider <slack|discord|generic> \
  --status success \
  --tool <tool-name> \
  --run-id <run-id> \
  --url-env <WEBHOOK_ENV_VAR> \
  --message "Run complete"
```

Supported providers:
- `generic` (JSON payload)
- `slack` (text payload)
- `discord` (text payload)

Exactly one of `--url`, `--url-env`, or `--secret-ref` is required.
Live HTTPS delivery requires trust roots via `--tls-ca-bundle` or `SSL_CERT_FILE` and fails fast when neither is provided.
`--dry-run` does not post to the webhook and does not require a CA bundle.

## Metadata payloads

Attach metadata with a JSON file:

```bash
notify send \
  --provider generic \
  --status failure \
  --tool <tool-name> \
  --run-id <run-id> \
  --url-env <WEBHOOK_ENV_VAR> \
  --meta /abs/path/to/metadata.json
```

The metadata file must contain a JSON object, which is attached as `meta`.

## Practical patterns

Wrap pipeline execution with success/failure notifications:

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="/path/to/config.yaml"
TOOL_NAME="tool"
TOOL_CLI="tool-cli"
RUN_ID="run-001"
WEBHOOK_ENV="NOTIFY_WEBHOOK"

if uv run "$TOOL_CLI" run -c "$CONFIG"; then
  notify send \
    --provider slack \
    --status success \
    --tool "$TOOL_NAME" \
    --run-id "$RUN_ID" \
    --url-env "$WEBHOOK_ENV" \
    --message "Run completed"
else
  notify send \
    --provider slack \
    --status failure \
    --tool "$TOOL_NAME" \
    --run-id "$RUN_ID" \
    --url-env "$WEBHOOK_ENV" \
    --message "Run failed"
  exit 1
fi
```

Send milestone notifications across stages:

```bash
notify send --provider generic --status started --tool <tool-name> --run-id <run-id> --url-env <WEBHOOK_ENV_VAR> \
  --message "Pipeline stage started"
notify send --provider generic --status running --tool <tool-name> --run-id <run-id> --url-env <WEBHOOK_ENV_VAR> \
  --message "Pipeline stage running"
notify send --provider generic --status success --tool <tool-name> --run-id <run-id> --url-env <WEBHOOK_ENV_VAR> \
  --message "Run finished"
```

## Canonical runbooks

- USR watcher onboarding and operations: [usr-events.md](usr-events.md)
- BU SCC batch deployment patterns: [BU SCC Batch + Notify runbook](../bu-scc/batch-notify.md)
- Workload-specific examples for this repository: [SGE HPC Ops workload reference](../bu-scc/sge-hpc-ops/references/workload-dnadesign.md)

## Documentation ownership

- Keep watcher semantics and setup flow in `docs/notify/usr-events.md`.
- Keep platform-specific submission examples in `docs/bu-scc/`.
- Keep `src/dnadesign/notify/README.md` as module-local package documentation.

## Dry run

Preview payloads without sending:

```bash
notify send \
  --provider slack \
  --status running \
  --tool <tool-name> \
  --run-id <run-id> \
  --url https://example.com/webhook \
  --dry-run
```
