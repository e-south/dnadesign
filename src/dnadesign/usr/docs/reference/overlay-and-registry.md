# USR overlay and registry contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-26


## Overlay merge semantics

USR overlays are append-only parts under `_derived/<namespace>/`.

On read/materialize, overlay view is deterministic last-writer-wins:

1. Overlay parts are ordered by `created_at` descending, then filename descending.
2. For each join key (`id` or `sequence`), newest value wins per column.

Operational implications:

- Join keys in a single overlay part must be unique.
- Re-attaching the same namespace/columns in a newer part overrides older values.
- Compact parts periodically with `uv run usr maintenance overlay-compact ...` to reduce read overhead.
- After compaction, future overlay-part appends are allowed; USR promotes the compact file into part form before appending.
- Compact overlay snapshots are not retained by default.
- Explicit overlay pruning is exposed through `uv run usr maintenance overlay-remove ...`; archive retention is bounded to the latest archived snapshot.

## Namespace registry (required)

All dataset mutations require a registry at the datasets root (`registry.yaml`).

- Register every namespace before attach/materialize.
- Reserved `usr_state` namespace must exist and is only mutated through the dedicated `uv run usr state ...` surface.
- First successful registration creates `registry.yaml` and includes `usr_state`.
- For repo-owned shared roots such as `src/dnadesign/usr/datasets`, `registry.yaml` is a tracked cross-tool contract for sibling tools that write or validate namespaced overlays.
- Keep shared-root registry changes committed and synced across clones before relying on `usr validate --strict`, `usr diff`, `usr pull`, or `usr push`.
- `usr:registry_hash` is derived from the serialized `registry.yaml` bytes, so canonical namespace and column ordering is part of the current contract.
- Overlay writers now also stamp `usr:namespace_contract_hash` for the specific namespace they emit.
- `usr:namespace_contract_hash` hashes only the namespace id plus ordered column name/type pairs; owner and description remain catalog metadata, not compatibility inputs.
- Opt into namespace-scoped validation explicitly with `--registry-mode namespace-current`, `namespace-frozen`, or `namespace-either`.

Shared sequence-product namespaces currently tracked in the repo registry:

- `usr_label`: canonical human-readable names and aliases
- `seq_annot`: imported GenBank-backed source annotation overlays
- `derived`: parent/child lineage, product kind, focal-selection, and feature-retention summaries

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
uv run usr maintenance registry-freeze densegen_demo
```

Auto-freeze behavior: on first dataset mutation with a registry present, USR writes `_registry/registry.<hash>.yaml` and stamps `usr:registry_hash` into `records.parquet`.

## Sequence-view sidecars

Semantic sequence views are stored outside overlay parts in dataset-local sidecars:

- path: `_views/sequence_views.parquet`
- cardinality: many semantic views may reference one base `records.parquet.id`
- authority: USR owns durable `view_id`, product kind, orientation, parent lineage, optional source intervals, and optional emitted anchor bounds
- mutability: human aliases may grow, but a `view_id` collision with different semantic content is an error

Conflict behavior is explicit-only:

- `error`: reject any conflicting existing semantic row
- `idempotent`: allow exact repeats
- `replace`: rewrite an existing view row
- `append_alias`: add human aliases only when the semantic key is unchanged

Sequence views are additive metadata. They do not replace sequence-derived base ids, and they are not inferred implicitly by downstream tools.

Merged anchor handoffs should use product kinds conservatively. For example,
`usr_prom_eth_cip_anchor` is a construct-ready promoter-insert handoff, so its
dataset-local sequence-view sidecar uses one `construct_insert` view per base
row with `context_kind=anchor_only` and `recommended_pooling=seq_mean`. Rows
that came from `construct_prom_eth_cip_reference_core60` keep
`analysis_only=true` and parent lineage, but the merged handoff does not
relabel every native or designed 60 bp row as `analysis_window`. The source
`construct_prom_eth_cip_reference_core60` dataset remains the authoritative
surface for true derived analysis-core products.

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
