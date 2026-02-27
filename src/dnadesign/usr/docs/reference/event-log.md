# USR event log contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


`.events.log` is newline-delimited JSON and is the operator integration boundary.

## Payload fields

Each event line includes:

- `event_version` (integer)
- `timestamp_utc` (RFC 3339 string)
- `action` (string)
- `dataset` (`name`, `root`)
- `args` (key-based secret redaction)
- `metrics` (object)
- `artifacts` (object)
- `maintenance` (object)
- `fingerprint` (`rows`, `cols`, `size_bytes`, optional `sha256` when `USR_EVENT_SHA256=1`)
- `registry_hash` (string or null)
- `actor` (`tool`, `run_id`, `host`, `pid`)
- `version` (USR package version)

Notify expects at minimum `event_version` and `action`.

## Integration boundary

- Notify consumes dataset `.events.log`.
- DenseGen runtime telemetry (`outputs/meta/events.jsonl`) is not Notify input.

See: [../../../../../docs/notify/usr-events.md](../../../../../docs/notify/usr-events.md)

## Next steps

- Sync and audit workflows: [../operations/sync-audit-loop.md](../operations/sync-audit-loop.md)
- Overlay and registry contract: [overlay-and-registry.md](overlay-and-registry.md)
