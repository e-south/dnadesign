## Notify Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

`notify` sends webhook notifications from Universal Sequence Record (USR) `.events.log` streams.
Start here for Notify setup, watch, and recovery. Use [Notify USR events operator manual](usr-events.md) when you need the full ordered procedure.

### Entry contract

- Audience: operators running watcher/replay loops, plus maintainers routing into package references.
- Prerequisites: workspace config and USR `.events.log`.
- Live delivery requires one webhook source.
- Reusable Slack/profile workflows use file-backed secret references (`--secret-source file` + `--secret-ref file://...`) with owner-only permissions (`chmod 600`).
- Explicit one-off or local-receiver drills can also use runtime overrides such as `--url` or `--url-env`.
- `notify ... --dry-run` can validate routing and event parsing without webhook URL resolution.
- Verify next: [notify profile doctor contract](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-profile-doctor).

### Choose a workflow

| Need | Start here | First command | Verify next |
| --- | --- | --- | --- |
| Start local watcher loops | [Notify USR events operator manual](usr-events.md) | `notify setup slack --tool <tool> --workspace <workspace> --secret-source file --secret-ref file://<abs-path-to-webhook-secret>` | `notify profile doctor --profile <profile.json>` |
| Follow a multi-source source-of-truth dataset into Notify and Infer | [Multi-source source-of-truth assembly](../../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md) | `notify setup resolve-events --tool construct --config "$CONSTRUCT_CONFIG" --json` | `notify usr-events watch --events "$USR_ROOT/$DOWNSTREAM_DATASET/.events.log" --dry-run --no-advance-cursor-on-dry-run` |
| Follow a construct-backed source-of-truth dataset into Notify and Infer | [Construct -> USR -> Infer source-of-truth runbook](../../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md) | `notify setup resolve-events --tool construct --config "$CONSTRUCT_CONFIG" --json` | `notify usr-events watch --events "$USR_ROOT/$DATASET_ID/.events.log" --dry-run --no-advance-cursor-on-dry-run` |
| Send one-off status messages | [notify send contract](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-send) | `notify send --status <status> --tool <tool> --run-id <id> --provider <provider> ...` | `notify send --dry-run ...` |
| Recover failed deliveries | [Recover flow](usr-events.md#recover-flow) | `notify spool drain --profile <profile.json>` | `notify spool drain --profile <profile.json> --fail-fast` |
| Run scheduler-managed Notify workflows | [BU SCC Batch + Notify runbook](../bu-scc/batch-notify.md) | follow scheduler runbook command chain | `notify profile doctor --profile <profile.json>` |
| Inspect internals and extension seams | [Notify package docs index](../../src/dnadesign/notify/docs/README.md) | open maintainer/reference docs first | [Maintainer architecture map](../../src/dnadesign/notify/docs/dev/architecture.md) |

### Start here

1. Run one setup/watch/recover workflow in [Notify USR events operator manual](usr-events.md).
2. Confirm strict mode rules in [Notify command contracts](../../src/dnadesign/notify/docs/reference/command-contracts.md).
3. Use [Multi-source source-of-truth assembly](../../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md) when upstream datasets are explicitly merged before construct writes the dataset that Notify should watch.
4. Use [Construct -> USR -> Infer source-of-truth runbook](../../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md) when Notify needs to observe a construct-backed consolidated dataset instead of a tool-local workspace loop.
5. When watcher validation is done and you need the downstream feature or learning step, continue to [Promoter characterization feature matrix](../../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md) or the [repository docs index](../README.md); Notify does not own that handoff.
6. Use [Notify package docs index](../../src/dnadesign/notify/docs/README.md) only when you need internals.
7. Use [BU SCC Batch + Notify runbook](../bu-scc/batch-notify.md) only for scheduler paths.

### Prompt-to-command router

| If the user asks... | Run this first | Then verify with |
| --- | --- | --- |
| "start a densegen workspace watcher and send to slack" | `notify setup slack --tool densegen --workspace <workspace> --secret-source file --secret-ref file://<abs-path-to-webhook-secret> --policy densegen` | `notify profile doctor --profile <config-dir>/outputs/notify/densegen/profile.json` |
| "start an infer workspace watcher and send to slack" | `notify setup slack --tool infer --workspace <workspace> --secret-source file --secret-ref file://<abs-path-to-webhook-secret> --policy infer` | `notify profile doctor --profile <config-dir>/outputs/notify/infer/profile.json` |
| "i already have a profile, just validate wiring" | `notify profile doctor --profile <profile.json>` | `notify usr-events watch --profile <profile.json> --dry-run` |
| "resume failed deliveries from spool" | `notify spool drain --profile <profile.json>` | `notify spool drain --profile <profile.json> --fail-fast` |
| "watch this workspace without passing profile path" | `notify usr-events watch --tool <tool> --workspace <workspace> --follow` | `notify setup resolve-events --tool <tool> --workspace <workspace>` |

### 2-minute operator path

```bash
uv run notify setup slack --tool densegen --workspace <workspace> --secret-source file --secret-ref file://<abs-path-to-webhook-secret> --policy densegen
uv run notify profile doctor --profile <config-dir>/outputs/notify/densegen/profile.json
uv run notify usr-events watch --tool densegen --workspace <workspace> --follow --wait-for-events
```

For the full quickstart (including webhook setup and dry-run checks), use [Minimal operator quickstart](usr-events.md#minimal-operator-quickstart).

### Interface contract summary

- Notify consumes USR `"<dataset>/.events.log"` only.
- DenseGen runtime telemetry (`outputs/meta/events.jsonl`) is not Notify input.
- Profile schema contract is [profile schema contract](../../src/dnadesign/notify/docs/reference/command-contracts.md#profile-schema-contract).
- `notify setup slack` mode contract: [notify setup slack](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-setup-slack).
- `notify usr-events watch` mode contract: [notify usr-events watch](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-usr-events-watch).
- No silent fallback: invalid mode/profile/secret inputs fail with explicit errors.

### Command surface map

| Command | Use when | Contract reference |
| --- | --- | --- |
| `notify send` | One-off notifications from scripts or jobs | [notify send](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-send) |
| `notify setup webhook` | Provision or resolve webhook secret refs | [notify setup webhook](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-setup-webhook) |
| `notify setup slack` | Build workspace-bound watcher profile | [notify setup slack](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-setup-slack) |
| `notify setup resolve-events` | Verify resolved `.events.log` path without writing profile | [notify setup resolve-events](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-setup-resolve-events) |
| `notify profile doctor` | Validate profile wiring and secret resolution | [notify profile doctor](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-profile-doctor) |
| `notify usr-events watch` | Stream `.events.log` to webhook | [notify usr-events watch](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-usr-events-watch) |
| `notify spool drain` | Retry failed deliveries from spool | [notify spool drain](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-spool-drain) |

### Troubleshooting and recovery

- Profile validation failures: run `notify profile doctor --profile <profile.json>` and resolve the first reported contract error.
- Events-source mismatch after workspace changes: rerun `notify setup slack --tool <tool> --workspace <workspace> --force`.
- HTTPS trust failures: provide `--tls-ca-bundle` or export `SSL_CERT_FILE`.
- Replay failures: run [Recover flow](usr-events.md#recover-flow).

### Runbooks

- Watcher onboarding and lifecycle: [Notify USR events operator manual](usr-events.md).
- Multi-source downstream handoff: [Multi-source source-of-truth assembly](../../src/dnadesign/usr/docs/operations/multi-source-source-of-truth-assembly.md).
- Construct-backed consolidated dataset handoff: [Construct -> USR -> Infer source-of-truth runbook](../../src/dnadesign/usr/docs/operations/construct-infer-source-of-truth-runbook.md).
- Downstream feature/learning branch after watcher validation: [Promoter characterization feature matrix](../../src/dnadesign/usr/docs/operations/promoter-characterization-feature-matrix.md).
- Scheduler workflows: [BU SCC Batch + Notify runbook](../bu-scc/batch-notify.md).
- Package docs index: [src/dnadesign/notify/docs/README.md](../../src/dnadesign/notify/docs/README.md).
- Reference index: [src/dnadesign/notify/docs/reference/README.md](../../src/dnadesign/notify/docs/reference/README.md).
