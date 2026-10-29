## Notify: consuming SeqLedger events

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-29

Use this runbook to set up, run, and recover `notify` watcher loops.
Notify consumes SeqLedger `.events.log` JSONL.

### Entry contract

- Audience: operators running local or scheduler-backed Notify watch loops.
- Prerequisites: workspace config and SeqLedger `.events.log`.
- Live delivery requires one webhook source.
- Reusable Slack/profile workflows use one file-backed webhook secret reference (`--secret-source file` + `--secret-ref file://...`).
- Explicit local-receiver drills can use runtime overrides such as `--url` or `--url-env`.
- `--dry-run` can validate event parsing and routing without webhook URL resolution.
- Verify next: `uv run notify profile doctor --profile <profile.json>` before live delivery.

### Minimal operator quickstart

```bash
# Resolver inputs.
WORKSPACE=<workspace>
CONFIG=<dnadesign_repo>/src/dnadesign/infer/workspaces/$WORKSPACE/config.yaml
PROFILE=<dnadesign_repo>/src/dnadesign/infer/workspaces/$WORKSPACE/outputs/notify/infer/profile.json

# Optional trust roots for shared compute environments.
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

# Confirm workspace names.
uv run notify setup list-workspaces --tool infer

# Create or resolve webhook secret reference from a local secret file.
WEBHOOK_SECRET_FILE=/abs/path/to/notify.webhook
touch "$WEBHOOK_SECRET_FILE"
chmod 600 "$WEBHOOK_SECRET_FILE"
WEBHOOK_REF="$(uv run notify setup webhook --secret-source file --secret-ref "file://$WEBHOOK_SECRET_FILE" --name inference-shared --json | python -c 'import json,sys; print(json.load(sys.stdin)["webhook"]["ref"])')"

# Create or refresh watcher profile.
uv run notify setup slack --tool infer --workspace "$WORKSPACE" --secret-ref "$WEBHOOK_REF" --secret-source file --policy infer

# Validate before watch.
uv run notify profile doctor --profile "$PROFILE"

# Dry-run payload mapping.
uv run notify seqledger-events watch --tool infer --workspace "$WORKSPACE" --dry-run --no-advance-cursor-on-dry-run

# Live watch loop.
uv run notify seqledger-events watch --tool infer --workspace "$WORKSPACE" --follow --wait-for-events --stop-on-terminal-status --idle-timeout 900
```

Follow-loop truncate behavior:
- Default `--on-truncate error` fails fast on rotation/truncation/disappearance while following.
- Use `--on-truncate restart` when log replacement should rewind and continue.
- The BU SCC watcher wrapper (`docs/bu-scc/jobs/notify-watch.qsub`) defaults to `NOTIFY_ON_TRUNCATE=restart`.

### Command contract: setup vs watch

- `notify setup ...`: creates/updates profile artifacts.
- `notify seqledger-events watch ...`: reads events and performs delivery.

Canonical command rules live in one place:
- [Notify command contracts](../../src/dnadesign/notify/docs/reference/command-contracts.md)

Fail-fast reminders:
- Setup fails when `--events` is mixed with resolver mode (`--tool` + one of `--workspace` or `--config`).
- Watch fails when resolver-mode profile is missing.
- Watch fails when profile `events_source` does not match resolver inputs.
- HTTPS delivery fails without `--tls-ca-bundle` or `SSL_CERT_FILE`.

### Setup flow

Resolver shorthand notes:
- Infer uses `--workspace <workspace-name>`.
- Construct uses `--workspace <workspace-name>` only when the workspace registry has one project, otherwise use `--workspace <workspace-name>:<project-id>`.
- Workspace shorthand for any tool is repo-rooted. Outside the repo checkout, set `DNADESIGN_REPO_ROOT=<repo-root>` or pass `--config` explicitly.
- External construct workspace roots can be exposed to Notify with `CONSTRUCT_WORKSPACE_ROOT`.

Default resolver-mode artifact paths:
- `<config-dir>/outputs/notify/<tool>/profile.json`
- `<config-dir>/outputs/notify/<tool>/cursor`
- `<config-dir>/outputs/notify/<tool>/spool/`

```bash
# Resolve expected events path without writing profile artifacts.
uv run notify setup resolve-events --tool infer --workspace "$WORKSPACE"

# Create profile from workspace resolver mode.
uv run notify setup slack --tool infer --workspace "$WORKSPACE" --secret-source file --secret-ref file://<abs-path-to-webhook-secret> --policy infer

# Create profile from explicit config path.
uv run notify setup slack --tool infer --config "$CONFIG" --secret-source file --secret-ref file://<abs-path-to-webhook-secret> --policy infer

# Example construct resolver inputs.
CONSTRUCT_CONFIG=/abs/path/to/construct/config.yaml
CONSTRUCT_WORKSPACE=demo_construct
CONSTRUCT_PROJECT=slot_a_window
# export DNADESIGN_REPO_ROOT=/abs/path/to/dnadesign
# export CONSTRUCT_WORKSPACE_ROOT=/abs/path/to/construct/workspaces

# Resolve a construct dataset from its config without guessing workspace project selection.
uv run notify setup resolve-events --tool construct --config "$CONSTRUCT_CONFIG" --json

# Optional construct workspace shorthand when the workspace is discoverable.
uv run notify setup resolve-events --tool construct --workspace "$CONSTRUCT_WORKSPACE:$CONSTRUCT_PROJECT" --json
```

Infer resolver reminder:
- `notify setup resolve-events --tool infer --config <config>` resolves either one legacy SeqLedger write-back
  destination or one sequence-view input dataset.
- Sequence-view Infer configs should be split into one operational dataset per watcher. This keeps
  Notify, resume state, and feature-sidecar writes aligned with a single SeqLedger `.events.log`.
- Multi-dataset Infer configs must use explicit `--events <path>` selection or be split into
  single-dataset runbooks; resolver mode fails fast rather than guessing which event stream owns the
  watcher.

### Run flow

```bash
# Dry-run from explicit profile.
uv run notify seqledger-events watch --profile "$PROFILE" --dry-run

# Live follow loop.
uv run notify seqledger-events watch --profile "$PROFILE" --follow --wait-for-events
```

Cold-start on an existing event stream:

If a profile already points at a long-lived `.events.log` and you want a new
watcher to consume only future events, seed the cursor to the current file size
before `--follow`:

```bash
EVENTS=/abs/path/to/dataset/.events.log
PROFILE=/abs/path/to/profile.json
CURSOR=/abs/path/to/cursor

mkdir -p "$(dirname "$CURSOR")"
stat -c %s "$EVENTS" > "$CURSOR"
uv run notify seqledger-events watch --profile "$PROFILE" --follow --idle-timeout 900
```

Do not preseed the cursor when replay is the goal.

Event-log gardening:

SeqLedger maintenance can archive an old `.events.log` and rewrite the live file to a
short operational tail. Treat that as an offline maintenance operation, not a
live Notify action. Stop watchers before gardening, run
`uv run seqledger maintenance event-log-garden <dataset> --write --acknowledge-notify-cursor-reset`,
then seed each Notify cursor to the new live `.events.log` size before
restarting follow mode. A cursor from the pre-garden file must not be reused.

Healthy watcher signals:

- watcher cursor advances
- watcher spool stays empty
- target dataset `.events.log` gains new events
- follow mode can briefly sit at the live tail while waiting for a newline-terminated event record; that is healthy and should not be treated as corruption
- Slack delivery can remain intentionally sparse between progress thresholds or
  heartbeat intervals

Cluster operations:
- BU SCC qsub workflow: [docs/bu-scc/runbooks/batch-notify.md](../bu-scc/runbooks/batch-notify.md)
- Submit-ready watcher job: [docs/bu-scc/jobs/notify-watch.qsub](../bu-scc/jobs/notify-watch.qsub)

### Recover flow

```bash
# Revalidate profile after workspace/config changes.
uv run notify profile doctor --profile "$PROFILE"

# JSON diagnostics for automation.
uv run notify profile doctor --profile "$PROFILE" --json

# Regenerate profile when events_source drift is detected.
uv run notify setup slack --tool infer --workspace "$WORKSPACE" --force

# Replay spooled payloads.
uv run notify spool drain --profile "$PROFILE"

# Stop at first replay error when debugging.
uv run notify spool drain --profile "$PROFILE" --fail-fast
```

### Common mistakes

- Using repository root config instead of workspace `config.yaml`.
- Mixing `--events` with resolver mode (`--tool` + `--workspace` or `--config`).
- Running live HTTPS delivery without trust roots.
- Sharing one cursor or spool path across unrelated runs.
- Starting a new live watcher on an old stream without seeding the cursor when
  replay is not desired.
- Expecting default policy filters to surface SeqLedger maintenance merge actions. If merge/carry operations must be visible, pass `--only-actions merge_datasets,attach,materialize` or set `NOTIFY_ACTIONS` explicitly.

### Infer progress semantics

`infer` progress messages are rendered from `args.infer_progress` in the SeqLedger event stream.

- `Families` reports persisted writeback coverage across rows still missing or stale for each feature family.
- On grouped infer `write_overlay_part` events, Notify can omit `Current output` because the event covers multiple columns at once; treat that update as aggregate writeback progress, not per-family GPU compute.
- A flat family percentage during a repair or resume pass can mean the current rows already had that family populated even while other families are still being repaired.
- Sequence-view feature lanes can set Infer `io.write_back: false`; those lanes
  emit sparse `infer_feature_bundle_progress` updates plus a terminal
  `infer_feature_bundle_complete` event on the watched SeqLedger stream. Low-level
  `infer_feature_*_write` sidecar events remain available for explicit audit
  filters but are not part of the default Slack policy.

### Event schema source of truth

- SeqLedger event contract: [SeqLedger event log reference](../../src/dnadesign/seqledger/docs/reference/event-log.md)

### Related docs

- Notify operations overview: [docs/notify/README.md](README.md)
- Notify command contracts: [src/dnadesign/notify/docs/reference/command-contracts.md](../../src/dnadesign/notify/docs/reference/command-contracts.md)
- Construct-backed consolidated dataset handoff: [Construct -> SeqLedger -> Infer shared dataset runbook](../../src/dnadesign/seqledger/docs/operations/assembly/construct-infer-shared-dataset-runbook.md)
