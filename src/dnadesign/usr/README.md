![Universal Sequence Record banner](assets/usr-banner.svg)

Universal Sequence Record (USR) provides canonical sequence datasets with explicit overlay and mutation-event contracts. It is the storage and integration boundary for tools that write sequence outputs and derived annotations. Use USR to maintain `records.parquet`, append namespaced overlays under `_derived/`, and emit `.events.log` mutation events for downstream operators.

## Documentation

- [USR docs index](docs/README.md): canonical map for all USR workflows and references.
- [CLI quickstart](docs/getting-started/cli-quickstart.md): first runnable path (`init` -> `import` -> `attach` -> `materialize` -> `export`).
- [Operations runbooks](docs/operations/README.md): remote sync, iterative batch loops, and transfer safety checks.
- [Reference index](docs/reference/README.md): stable contracts for schema, overlays/registry, event payloads, and API usage.
- [Architecture introspection](docs/architecture-introspection.md): package lifecycle, component boundaries, and interaction map.
- [Repository docs index](../../../docs/README.md): cross-tool workflows that connect USR with DenseGen, Infer, and Notify.
