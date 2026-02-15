# Notify

Notify reads Universal Sequence Record mutation events and posts selected events to webhook providers.

## Contents
- [At a glance](#at-a-glance)
- [Fast operator path](#fast-operator-path)
- [Read order](#read-order)
- [Maintainer code map](#maintainer-code-map)
- [Key boundary](#key-boundary)
- [Observer Contract](#observer-contract)

## At a glance

Use Notify when you want:
- restart-safe event watching with cursor offsets
- action/tool filtering before delivery
- spool-and-drain behavior for unstable network environments

Do not use Notify for:
- DenseGen runtime telemetry (`outputs/meta/events.jsonl`)
- generic log shipping

Contract:
- input stream is Universal Sequence Record `<dataset>/.events.log` newline-delimited JSON
- Notify treats Universal Sequence Record events as an external, versioned contract
- without `--profile`, choose exactly one webhook source:
  - `--url`
  - `--url-env`
  - `--secret-ref`

Default profile privacy is strict:
- `include_args=false`
- `include_context=false`
- `include_raw_event=false`

Artifact placement default:
- keep Notify artifacts with the run workspace being watched (relative to the tool config directory):
  - `<config-dir>/outputs/notify/<tool>/profile.json`
  - `<config-dir>/outputs/notify/<tool>/cursor`
  - `<config-dir>/outputs/notify/<tool>/spool/`

## Fast operator path

```bash
WORKSPACE=<workspace>
CONFIG=<dnadesign_repo>/src/dnadesign/densegen/workspaces/$WORKSPACE/config.yaml

# Discover available workspace names for your tool.
uv run notify setup list-workspaces --tool densegen

# One-time setup by workspace shorthand (stores webhook securely and writes profile under <config-dir>/outputs/notify/<tool>/).
uv run notify setup slack --tool densegen --workspace "$WORKSPACE" --secret-source auto
# Explicit path fallback:
# uv run notify setup slack --tool densegen --config "$CONFIG" --secret-source auto

# Validate the resolved profile.
uv run notify profile doctor --profile "src/dnadesign/densegen/workspaces/$WORKSPACE/outputs/notify/densegen/profile.json"

# Preview payloads without posting (autoload profile from --tool/--workspace).
uv run notify usr-events watch --tool densegen --workspace "$WORKSPACE" --dry-run

# Run live watcher (same autoload mode).
uv run notify usr-events watch --tool densegen --workspace "$WORKSPACE" --follow

# Retry failed payloads from spool.
uv run notify spool drain --profile "src/dnadesign/densegen/workspaces/$WORKSPACE/outputs/notify/densegen/profile.json"
```

## Read order

- Canonical operators runbook: [docs/notify/usr_events.md](../../../docs/notify/usr_events.md)
- Module-local pointer page: [docs/usr_events.md](docs/usr_events.md)
- Command anatomy: [notify setup slack flags and expectations](../../../docs/notify/usr_events.md#command-anatomy-notify-setup-slack)
- Setup onboarding: [Slack setup onboarding](../../../docs/notify/usr_events.md#slack-setup-onboarding-3-minutes)
- End-to-end stack demo: [DenseGen -> Universal Sequence Record -> Notify demo](../densegen/docs/demo/demo_usr_notify.md)
- Universal Sequence Record event schema source: [Universal Sequence Record event log schema](../usr/README.md#event-log-schema)

## Maintainer code map

Runtime and option resolution:
- `src/dnadesign/notify/cli.py`: Typer command surface and handler wiring.
- `src/dnadesign/notify/cli_runtime.py`: watch and spool runtime execution logic.
- `src/dnadesign/notify/cli_resolve.py`: profile and path resolution helpers.
- `src/dnadesign/notify/profile_flows.py`: setup and wizard profile construction.

Tool and workflow extension points:
- `src/dnadesign/notify/events_source.py`: tool resolver registry.
- `src/dnadesign/notify/events_source_builtin.py`: built-in resolver installs (`densegen`, `infer_evo2`).
- `src/dnadesign/notify/workspace_source.py`: workspace-name to config-path resolver registry.
- `src/dnadesign/notify/tool_events.py`: tool-event override and evaluator registry.
- `src/dnadesign/notify/tool_event_packs_builtin.py`: built-in tool-event pack installs.
- `src/dnadesign/notify/workflow_policy.py`: policy defaults and profile-path defaults.

## Key boundary

Notify reads:
- `<usr_root>/<dataset>/.events.log` (Universal Sequence Record event stream)

Notify does not read:
- `densegen/.../outputs/meta/events.jsonl`

## Observer Contract

Notify is an observer control plane:
- it does not launch tool runs
- it does not mutate tool command-line arguments, environment variables, or runtime behavior
- it only resolves where Universal Sequence Record events are expected and watches that stream

Tool integration contract:
- `notify setup slack --tool <tool> --config <workspace-config.yaml>` resolves expected Universal Sequence Record `.events.log` destination
- `notify setup slack --tool <tool> --workspace <workspace-name>` resolves config path from tool workspace root and then resolves expected Universal Sequence Record `.events.log` destination
- `notify setup list-workspaces --tool <tool>` lists discoverable workspace names for shorthand mode
- `notify usr-events watch --tool <tool> --config <workspace-config.yaml>` auto-loads profile from `<config-dir>/outputs/notify/<tool>/profile.json`
- `notify usr-events watch --tool <tool> --workspace <workspace-name>` uses workspace shorthand for the same auto-profile flow
- auto-profile mode is fail-fast: if profile `events_source` exists and does not match `--tool/--config` or `--tool/--workspace`, watch exits with a mismatch error
- `notify setup resolve-events --tool <tool> --config <config.yaml>` resolves events path/policy without writing a profile
- supported resolvers: `densegen`, `infer_evo2`
- profile stores `events_source` metadata (`tool`, `config`) so watcher restarts can re-resolve paths and avoid stale bindings
- unsupported tools fail fast with explicit errors (no implicit tool fallback)

Webhook contract:
- secure mode `--secret-source auto|keychain|secretservice`
- env mode supports `--url-env <ENV_VAR>` and defaults to `NOTIFY_WEBHOOK` when omitted
- Python keyring support is included in the project lockfile; Notify uses it first and falls back to OS commands (`security`/`secret-tool`) when needed
