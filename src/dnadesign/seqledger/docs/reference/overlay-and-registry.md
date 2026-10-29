# SeqLedger overlay and registry contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28


## Overlay merge semantics

SeqLedger overlays are append-only parts under `_derived/<namespace>/`.

On read/materialize, overlay view is deterministic last-writer-wins:

1. Overlay parts are ordered by `created_at` descending, then filename descending.
2. For each join key (`id` or `sequence`), newest value wins per column.

Operational implications:

- Join keys in a single overlay part must be unique.
- Re-attaching the same namespace/columns in a newer part overrides older values.
- Compact parts periodically with `uv run seqledger maintenance overlay-compact ...` to reduce read overhead.
- Refresh compact overlay metadata with `uv run seqledger maintenance overlay-refresh-metadata ...` when the namespace contract is current but the full-registry hash stamp is stale.
- After compaction, future overlay-part appends are allowed; SeqLedger promotes the compact file into part form before appending.
- Compact overlay snapshots are not retained by default.
- Explicit overlay pruning is exposed through `uv run seqledger maintenance overlay-remove ...`; archive retention is bounded to the latest archived snapshot.

## Namespace registry (required)

All dataset mutations require a registry at the datasets root (`registry.yaml`).

- Register every namespace before attach/materialize.
- Reserved `usr_state` namespace must exist and is only mutated through the dedicated `uv run seqledger state ...` surface.
- First successful registration creates `registry.yaml` and includes `usr_state`.
- For repo-owned shared roots such as `src/dnadesign/seqledger/datasets`, `registry.yaml` is a tracked cross-tool contract for sibling tools that write or validate namespaced overlays.
- Keep shared-root registry changes committed and synced across clones before relying on `seqledger validate --strict`, `seqledger diff`, `seqledger pull`, or `seqledger push`.
- `usr:registry_hash` is derived from the serialized `registry.yaml` bytes, so canonical namespace and column ordering is part of the current contract.
- Overlay writers now also stamp `usr:namespace_contract_hash` for the specific namespace they emit.
- `usr:namespace_contract_hash` hashes only the namespace id plus ordered column name/type pairs; owner and description remain catalog metadata, not compatibility inputs.
- Opt into namespace-scoped validation explicitly with `--registry-mode namespace-current`, `namespace-frozen`, or `namespace-either`.

Default strict validation uses the full current registry hash. That is intentionally conservative for shared dataset roots because a registry-wide edit can affect any downstream writer. When default strict validation fails on an overlay hash but namespace-scoped validation passes, the overlay rows still satisfy the namespace they use:

```bash
uv run seqledger validate <dataset> --strict --registry-mode namespace-current
```

For compact overlays, restamp the metadata through the package-owned maintenance surface:

```bash
uv run seqledger maintenance overlay-refresh-metadata <dataset> --namespace <namespace>
```

For part-directory overlays, compact instead:

```bash
uv run seqledger maintenance overlay-compact <dataset> --namespace <namespace>
```

Shared sequence-product namespaces currently tracked in the repo registry:

- `usr_label`: canonical human-readable names and aliases
- `seq_annot`: imported GenBank-backed source annotation overlays
- `derived`: parent/child lineage, product kind, focal-selection, and feature-retention summaries

Register namespace:

```bash
uv run seqledger namespace register mock \
  --columns 'mock__score:float64,mock__vec:list<float64>' \
  --owner "your-name" \
  --description "example derived metrics"
```

Inspect registry:

```bash
uv run seqledger namespace list
uv run seqledger namespace show mock
```

Freeze registry snapshot into dataset:

```bash
uv run seqledger maintenance registry-freeze densegen_demo
```

Auto-freeze behavior: on first dataset mutation with a registry present, SeqLedger writes `_registry/registry.<hash>.yaml` and stamps `usr:registry_hash` into `records.parquet`.

## Sequence-view sidecars

Semantic sequence views are stored outside overlay parts in dataset-local sidecars:

- path: `_views/sequence_views.parquet`
- cardinality: many semantic views may reference one base `records.parquet.id`
- authority: SeqLedger owns durable `view_id`, product kind, orientation, parent lineage, optional source intervals, and optional emitted anchor bounds
- mutability: human aliases may grow, but a `view_id` collision with different semantic content is an error

Conflict behavior is explicit-only:

- `error`: reject any conflicting existing semantic row
- `idempotent`: allow exact repeats without rewriting the sidecar or emitting a write event
- `replace`: rewrite an existing view row
- `append_alias`: add human aliases only when the semantic key is unchanged

Sequence-view and view-semantics writers return the number of distinct rows
changed. An unchanged request returns zero; mixed batches count only changed rows.

Sequence views are additive metadata. They do not replace sequence-derived base ids, and they are not inferred implicitly by downstream tools.

## Design contracts

- Canonical essentials are stable: `id`, `bio_type`, `sequence`, `alphabet`, `length`, `source`, `created_at`.
- Base records are append-only; updates happen via overlays; base rewrites are maintenance operations.
- Writes are atomic; snapshots are written under `_snapshots/`.
- Tombstones are logical (`usr__deleted`, `usr__deleted_at`, `usr__deleted_reason`) and hidden by default.
- `usr_state` fields are standardized and registry-governed.

Explicit propagation rule for downstream tools:

- carry a namespace unchanged only when the downstream dataset can preserve the same key and semantics
- project a namespace only with an explicit coordinate or id transform
- summarize rich source metadata into downstream overlays when exact carry-through would be misleading
- drop a namespace intentionally when the downstream product no longer supports it

Construct and Infer must not assume rich overlays or sequence-view semantics propagate automatically through merge/materialize paths.

## Next steps

- Schema definitions: [schema-contract.md](schema-contract.md)
- Event payload contract: [event-log.md](event-log.md)
