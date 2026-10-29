<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../README.md)

# SeqLedger event log contract

`.events.log` records dataset mutations as newline-delimited JSON. Event consumers can use it to follow changes without inspecting every data file.
Each encoded record, including its terminal newline, is limited to 1 MiB. Store
large data in an artifact and reference it from the event instead of embedding it
in the ledger.

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
- `fingerprint` (`rows`, `cols`, `size_bytes`, optional `sha256` when `SEQLEDGER_EVENT_SHA256=1`)
- `registry_hash` (string or null)
- `actor` (`tool`, `run_id`, `host`, `pid`)
- `version` (SeqLedger package version)

Notify expects at minimum `event_version` and `action`.

## Integration boundary

[Notify](../../../../../docs/notify/seqledger-events.md) consumes dataset `.events.log`. A producing tool's separate runtime log is not interchangeable with this mutation stream.

## Next steps

- Sync and audit workflows: [SeqLedger sync audit loop](../operations/sync/audit-loop.md)
- Overlay and registry contract: [overlay-and-registry.md](overlay-and-registry.md)
