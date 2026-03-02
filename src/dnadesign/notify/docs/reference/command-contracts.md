## Notify command contracts

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-01

This page is the tool-local source for Notify command invocation contracts and fail-fast behavior.

### notify setup webhook

- Purpose: provision or resolve reusable webhook secret references.
- Requires one webhook source path via secret source options.
- Does not require tool/workspace/config/profile inputs.
- Emits machine-readable output with `--json`.

### notify setup list-workspaces

- Purpose: list available workspace names for a resolver-mode tool.
- Requires `--tool <name>`.
- Emits one workspace name per line, or machine-readable output with `--json`.

### notify setup resolve-events

- Purpose: resolve expected USR `.events.log` path and default policy without writing profile artifacts.
- Requires resolver mode with `--tool` and exactly one of `--config` or `--workspace`.
- `--print-policy` and `--json` are mutually exclusive.
- Fails fast on invalid resolver inputs and unknown workspace/config paths.

### notify setup slack

- Mode contract:
  - explicit mode: `--events <path>`.
  - resolver mode: `--tool` with exactly one of `--config` or `--workspace`.
- Mixed explicit and resolver flags fail fast.
- Resolver mode records `events_source` and uses policy defaults when `--policy` is omitted.
- Profile path defaults to `<config-dir>/outputs/notify/<tool>/profile.json` in resolver mode.

### notify send

- Requires `--status`, `--tool`, `--run-id`, and `--provider`.
- Requires exactly one webhook source: `--url`, `--url-env`, or `--secret-ref`.
- `--dry-run` prints provider-formatted payload and exits before HTTP post.
- `--dry-run` still enforces webhook source and provider URL validation contracts.
- HTTPS webhook delivery requires trust roots via `--tls-ca-bundle` or `SSL_CERT_FILE`.

### notify usr-events watch

- Invocation contract: pass exactly one mode family.
  - `--profile <profile.json>`
  - `--events <path>`
  - resolver mode: `--tool` with exactly one of `--config` or `--workspace`
- `--config/--workspace` cannot be combined with `--profile` or `--events`.
- Resolver mode fails when the expected auto-profile file is missing and prints setup guidance.
- If profile `events_source` disagrees with current resolver inputs, watch fails fast with mismatch details.
- Live delivery source contract without profile webhook values: exactly one of `--url`, `--url-env`, or `--secret-ref`.
- `--dry-run` bypasses webhook posting and can run without webhook URL resolution.

### notify profile doctor

- Requires `--profile`.
- Validates profile JSON schema, webhook source fields, event-source metadata, and version invariants.

### notify spool drain

- Replays failed payloads from spool directory or profile-resolved spool path.
- Supports `--fail-fast` to stop on first failed replay.
- Delivery source overrides follow the same webhook source contract used by watch/send.

### profile schema contract

- `profile_version` must be `2`.
- Profile files must not store plaintext `url` fields.
- Required string fields and webhook/event source structures are validated at load.
- Non-generic policies require explicit `only_actions` and `only_tools` in profile payload.

### observer boundary

- Notify reads USR `"<dataset>/.events.log"` JSONL streams.
- Notify does not consume DenseGen runtime telemetry (`outputs/meta/events.jsonl`).
- Notify resolves observation paths and delivery wiring; it does not launch upstream tool runs.

### no-silent-fallback contract

- Invalid resolver-mode combinations fail immediately with explicit CLI guidance.
- Missing auto-generated resolver-mode profile files fail with actionable setup command hints.
- Invalid webhook source combinations and TLS trust configuration fail before delivery attempts.
- Profile schema/version mismatches fail before watch and spool runtime loops.

### Runtime evidence pointers

- watch mode exclusivity: `src/dnadesign/notify/runtime/watch_runner_contract.py`
- setup mode exclusivity: `src/dnadesign/notify/profiles/flow_events.py`
- setup resolver helper contract: `src/dnadesign/notify/cli/handlers/setup/resolve_events_cmd.py`
- webhook source and TLS contracts: `src/dnadesign/notify/delivery/validation.py`
- send command runtime path: `src/dnadesign/notify/cli/handlers/send.py`
- spool replay contract and fail-fast option: `src/dnadesign/notify/runtime/spool_runner.py`
- auto-profile and events-source mismatch behavior: `src/dnadesign/notify/runtime/watch_runner_resolution.py`
- profile schema/version invariants: `src/dnadesign/notify/profiles/schema/reader.py`
