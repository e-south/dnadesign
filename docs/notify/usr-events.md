## Notify: consuming Universal Sequence Record events

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28

Notify consumes Universal Sequence Record mutation events from `.events.log` JSONL files and sends selected events to webhook providers.
The integration contract is Universal Sequence Record `.events.log` only; DenseGen runtime diagnostics (`outputs/meta/events.jsonl`) are out of scope.

### Minimal operator quickstart

```bash
WORKSPACE=<workspace>
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/$WORKSPACE/config.yaml
PROFILE=<dnadesign_repo>/src/dnadesign/densegen/workspaces/$WORKSPACE/outputs/notify/densegen/profile.json

# Shared computing cluster certificate trust chain for secure webhook delivery.
export SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem

# 1) List workspace names for shorthand mode.
uv run notify setup list-workspaces --tool densegen

# 2) Configure or reuse webhook secret ref.
WEBHOOK_REF="$(uv run notify setup webhook --secret-source auto --name densegen-shared --json | python -c 'import json,sys; print(json.load(sys.stdin)["webhook"]["ref"])')"

# 3) Build profile from tool+workspace resolver.
uv run notify setup slack \
  --tool densegen \
  --workspace "$WORKSPACE" \
  --secret-ref "$WEBHOOK_REF" \
  --secret-source auto \
  --policy densegen

# 4) Validate profile wiring.
uv run notify profile doctor --profile "$PROFILE"

# 5) Preview payload mapping without posting.
uv run notify usr-events watch \
  --tool densegen \
  --workspace "$WORKSPACE" \
  --dry-run \
  --no-advance-cursor-on-dry-run

# 6) Run live watcher.
uv run notify usr-events watch \
  --tool densegen \
  --workspace "$WORKSPACE" \
  --follow \
  --wait-for-events \
  --stop-on-terminal-status \
  --idle-timeout 900

# 7) Drain failed payloads from spool.
uv run notify spool drain --profile "$PROFILE"
```

### Command contract: setup vs watch

`notify setup ...` commands are profile/secret orchestration commands.
They resolve and write watcher configuration but do not watch events.

`notify usr-events watch ...` is the runtime event-loop command.
It reads `.events.log`, applies filters/policies, posts payloads, and maintains cursor/spool state.

No-silent-fallback rules:
- Setup fails when resolver inputs are ambiguous (`--events` mixed with `--tool` resolver mode).
- Watch fails when profile `events_source` disagrees with supplied `--tool/--config` or `--tool/--workspace`.
- HTTPS delivery fails without trust roots (`--tls-ca-bundle` or `SSL_CERT_FILE`).

### Setup flow

Artifact placement (default resolver mode):
- `<config-dir>/outputs/notify/<tool>/profile.json`
- `<config-dir>/outputs/notify/<tool>/cursor`
- `<config-dir>/outputs/notify/<tool>/spool/`

#### Command anatomy: `notify setup webhook`

Use this command to provision or resolve a reusable webhook secret reference.
It does not require tool/workspace/config/profile parameters.

```bash
uv run notify setup webhook --secret-source auto --name densegen-shared --json
```

Outputs:
- JSON with `ok`, `name`, and `webhook.ref` when `--json` is set
- secure ref source (`env`, `secret_ref`) and reference value

Common options:
- `--secret-source auto|env|keychain|secretservice|file`
- `--name <logical-name>`
- `--secret-ref <backend://...>`
- `--webhook-url <url>`
- `--store-webhook/--no-store-webhook`

#### Command anatomy: `notify setup slack`

Use this command to build or refresh a watcher profile.
Pass either explicit events path mode or resolver mode.

Resolver mode (recommended):
```bash
uv run notify setup slack \
  --tool densegen \
  --workspace "$WORKSPACE" \
  --secret-source auto \
  --policy densegen
```

Equivalent explicit-config mode:
```bash
uv run notify setup slack \
  --tool densegen \
  --config "$CONFIG" \
  --secret-source auto \
  --policy densegen
```

Flags that shape contracts:
- `--tool` with exactly one of `--config` or `--workspace` for resolver mode
- `--events /abs/path/to/.events.log` for explicit mode
- `--profile`, `--cursor`, `--spool-dir` optional overrides
- `--policy densegen|infer_evo2|generic`
- `--force` required to overwrite existing profile

Profile expectations:
- `profile_version` must be `2`
- profile stores webhook references, not plaintext webhook URLs
- resolver mode stores `events_source` metadata (`tool`, `config`) for drift-safe restarts

### Run flow

Dry-run before live posting:

```bash
uv run notify usr-events watch --tool densegen --workspace "$WORKSPACE" --dry-run --no-advance-cursor-on-dry-run
```

Live watcher loop:

```bash
uv run notify usr-events watch \
  --tool densegen \
  --workspace "$WORKSPACE" \
  --follow \
  --wait-for-events \
  --stop-on-terminal-status \
  --idle-timeout 900
```

Batch mode reference:
- BU SCC qsub workflow: [docs/bu-scc/batch-notify.md](../bu-scc/batch-notify.md)
- Submit-ready watcher script: [docs/bu-scc/jobs/notify-watch.qsub](../bu-scc/jobs/notify-watch.qsub)

### Recover flow

Profile validation:

```bash
uv run notify profile doctor --profile "$PROFILE"
uv run notify profile doctor --profile "$PROFILE" --json
```

Spool replay:

```bash
uv run notify spool drain --profile "$PROFILE"
```

Regenerate stale profile bindings:

```bash
uv run notify setup slack --tool densegen --workspace "$WORKSPACE" --force
```

### Common mistakes

- Passing repository root config instead of workspace `config.yaml` in resolver mode.
- Mixing `--events` with `--tool/--config` or `--tool/--workspace`.
- Running live HTTPS mode without CA trust roots configured.
- Expecting Notify to consume DenseGen runtime telemetry (`outputs/meta/events.jsonl`).
- Sharing one cursor/spool path across unrelated runs.

### Required event fields (minimum contract)

Each USR event line must be a JSON object with:
- `event_version` integer
- `timestamp` string
- `action` string
- `entity.id` string
- `actor.tool` string

Notify validates this event contract before transformation/delivery.

### Related docs

- Notify docs index: [docs/notify/README.md](README.md)
- Module-local quick operations page: [src/dnadesign/notify/docs/usr-events.md](../../src/dnadesign/notify/docs/usr-events.md)
- DenseGen integration walkthrough: [DenseGen -> USR -> Notify tutorial](../../src/dnadesign/densegen/docs/tutorials/demo_usr_notify.md)
- Universal Sequence Record event schema source: [USR event log reference](../../src/dnadesign/usr/docs/reference/event-log.md)
