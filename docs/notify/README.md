## Notify Operations

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

`notify` sends webhook notifications from Universal Sequence Record (USR) `.events.log` streams.
Start here for setup, watching, and recovery. Use [Notify USR events operator manual](usr-events.md) when you want the full ordered procedure.
If you already have a workspace and a webhook secret file, jump to [Quick path](#quick-path). Use the workflow table when you need a specific handoff or recovery route.

### Before you start

- Prerequisites: workspace config and USR `.events.log`.
- Live delivery requires one webhook source.
- Reusable Slack/profile workflows use file-backed secret references (`--secret-source file` + `--secret-ref file://...`) with owner-only permissions (`chmod 600`).
- Explicit one-off or local-receiver drills can also use runtime overrides such as `--url` or `--url-env`.
- `notify ... --dry-run` can validate routing and event parsing without webhook URL resolution.
- Notify reads USR `.events.log`; DenseGen runtime telemetry (`outputs/meta/events.jsonl`) is not Notify input.
- When starting a live watcher on an existing stream with a materialized
  profile but no cursor file yet, seed the cursor to the current `.events.log`
  size first unless replay is intentional.
- Verify next with [notify profile doctor contract](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-profile-doctor).

### Choose a workflow

| Need | Start here | First command | Verify next |
| --- | --- | --- | --- |
| Start local watcher loops | [Notify USR events operator manual](usr-events.md) | `notify setup slack --tool <tool> --workspace <workspace> --secret-source file --secret-ref file://<abs-path-to-webhook-secret>` | `notify profile doctor --profile <profile.json>` |
| Follow a multi-source shared dataset into Notify and Infer | [Multi-source shared dataset assembly](../../src/dnadesign/usr/docs/operations/assembly/multi-source-shared-dataset.md) | `notify setup resolve-events --tool construct --config "$CONSTRUCT_CONFIG" --json` | `notify usr-events watch --events "$USR_ROOT/$DOWNSTREAM_DATASET/.events.log" --dry-run --no-advance-cursor-on-dry-run` |
| Follow a construct-backed shared dataset into Notify and Infer | [Construct -> USR -> Infer shared dataset runbook](../../src/dnadesign/usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md) | `notify setup resolve-events --tool construct --config "$CONSTRUCT_CONFIG" --json` | `notify usr-events watch --events "$USR_ROOT/$DATASET_ID/.events.log" --dry-run --no-advance-cursor-on-dry-run` |
| Send one-off status messages | [notify send contract](../../src/dnadesign/notify/docs/reference/command-contracts.md#notify-send) | `notify send --status <status> --tool <tool> --run-id <id> --provider <provider> ...` | `notify send --dry-run ...` |
| Recover failed deliveries | [Recover flow](usr-events.md#recover-flow) | `notify spool drain --profile <profile.json>` | `notify spool drain --profile <profile.json> --fail-fast` |
| Run scheduler-managed Notify workflows | [BU SCC Batch + Notify runbook](../bu-scc/runbooks/batch-notify.md) | follow scheduler runbook command chain | `notify profile doctor --profile <profile.json>` |
| Inspect internals and extension seams | [Notify package docs index](../../src/dnadesign/notify/docs/README.md) | read the maintainer architecture map | [Maintainer architecture map](../../src/dnadesign/notify/docs/dev/architecture.md) |

### Quick path

```bash
uv run notify setup slack --tool densegen --workspace <workspace> --secret-source file --secret-ref file://<abs-path-to-webhook-secret> --policy densegen
uv run notify profile doctor --profile <config-dir>/outputs/notify/densegen/profile.json
uv run notify usr-events watch --tool densegen --workspace <workspace> --follow --wait-for-events
```

For the full quickstart (including webhook setup and dry-run checks), use [Minimal operator quickstart](usr-events.md#minimal-operator-quickstart).

For live Infer study lanes, prefer one watcher per lane config and one watcher
per destination dataset. Do not use one live watcher for a multi-destination
Infer config when Notify and resume posture matter.

### Troubleshooting

- Profile validation failures: run `notify profile doctor --profile <profile.json>` and resolve the first reported contract error.
- Events-source mismatch after workspace changes: rerun `notify setup slack --tool <tool> --workspace <workspace> --force`.
- HTTPS trust failures: provide `--tls-ca-bundle` or export `SSL_CERT_FILE`.
- Quiet watcher stdout is not a failure by itself. Infer notify policies emit
  sparse running updates; check cursor movement and spool backlog before
  assuming delivery is broken.
- Replay failures: run [Recover flow](usr-events.md#recover-flow).

### References

- Watcher onboarding and lifecycle: [Notify USR events operator manual](usr-events.md).
- Command contracts: [Notify command contracts](../../src/dnadesign/notify/docs/reference/command-contracts.md).
- Multi-source downstream handoff: [Multi-source shared dataset assembly](../../src/dnadesign/usr/docs/operations/assembly/multi-source-shared-dataset.md).
- Construct-backed consolidated dataset handoff: [Construct -> USR -> Infer shared dataset runbook](../../src/dnadesign/usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md).
- Downstream feature and learning steps after watcher validation: [Promoter characterization feature matrix](../../src/dnadesign/usr/docs/operations/promoter/characterization-feature-matrix.md).
- Scheduler workflows: [BU SCC Batch + Notify runbook](../bu-scc/runbooks/batch-notify.md).
- Package docs index: [src/dnadesign/notify/docs/README.md](../../src/dnadesign/notify/docs/README.md).
