# USR docs

## At a glance

**Intent:** USR is the canonical, auditable sequence store and mutation/event boundary for `dnadesign`.

**When to use:**
- Store canonical sequence datasets (generated and curated).
- Attach derived metrics or annotations without rewriting base records.
- Emit mutation events for operators and automation (Notify).
- Govern derived columns with explicit namespace registry contracts.

**When not to use:**
- Not a sequence generator (use DenseGen and related tools).
- Not an alerting transport (use Notify).

**Boundary / contracts:**
- `.events.log` is the integration boundary; Notify consumes this stream.
- Derived columns must be namespaced as `<namespace>__<field>`.
- Namespaces must be registered before attach/materialize operations.

**Start here:**
- `../README.md` (concepts + CLI quickstart)
- `../README.md#namespace-registry-required`
- `../README.md#event-log-schema`
- `operations/sync.md` (remote sync)

## Start here

- USR concepts plus CLI quickstart: `../README.md`
- Overlay plus registry contract: `../README.md#namespace-registry-required`
- Overlay merge semantics: `../README.md#how-overlays-merge-conflict-resolution`
- Event log schema (Notify input): `../README.md#event-log-schema`

## Integration

- DenseGen writes into USR via the `densegen` namespace:
  - DenseGen outputs reference: `../../densegen/docs/reference/outputs.md`
- Notify consumes USR `.events.log`:
  - Notify operators doc: `../../../../docs/notify/usr_events.md`

## Operations

- Remote sync: `operations/sync.md`
- Dev notes: `dev/journal.md`
