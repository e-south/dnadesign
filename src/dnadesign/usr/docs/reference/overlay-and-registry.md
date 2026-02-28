# USR overlay and registry contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


## Overlay merge semantics

USR overlays are append-only parts under `_derived/<namespace>/`.

On read/materialize, overlay view is deterministic last-writer-wins:

1. Overlay parts are ordered by `created_at` descending, then filename descending.
2. For each join key (`id` or `sequence`), newest value wins per column.

Operational implications:

- Join keys in a single overlay part must be unique.
- Re-attaching the same namespace/columns in a newer part overrides older values.
- Compact parts periodically with `uv run usr maintenance overlay-compact ...` to reduce read overhead.

## Namespace registry (required)

All dataset mutations require a registry at the datasets root (`registry.yaml`).

- Register every namespace before attach/materialize.
- Reserved `usr_state` namespace must exist and remain unmodified.
- First successful registration creates `registry.yaml` and includes `usr_state`.

Register namespace:

```bash
uv run usr namespace register mock \
  --columns 'mock__score:float64,mock__vec:list<float64>' \
  --owner "your-name" \
  --description "example derived metrics"
```

Inspect registry:

```bash
uv run usr namespace list
uv run usr namespace show mock
```

Freeze registry snapshot into dataset:

```bash
uv run usr maintenance registry-freeze densegen/demo
```

Auto-freeze behavior: on first dataset mutation with a registry present, USR writes `_registry/registry.<hash>.yaml` and stamps `usr:registry_hash` into `records.parquet`.

## Design contracts

- Canonical essentials are stable: `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`, `created_at`.
- Base records are append-only; updates happen via overlays; base rewrites are maintenance operations.
- Writes are atomic; snapshots are written under `_snapshots/`.
- Tombstones are logical (`usr__deleted`, `usr__deleted_at`, `usr__deleted_reason`) and hidden by default.
- `usr_state` fields are standardized and registry-governed.

## Next steps

- Schema definitions: [schema-contract.md](schema-contract.md)
- Event payload contract: [event-log.md](event-log.md)
