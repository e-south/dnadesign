![SeqLedger banner](assets/seqledger-banner.svg)

SeqLedger stores sequence datasets with explicit overlay and mutation-event contracts. It is the integration boundary for tools that write sequence outputs and derived annotations, maintain `records.parquet`, append namespaced overlays under `_derived/`, and emit `.events.log` mutation events for downstream operators.

## Documentation

- [SeqLedger docs index](docs/README.md): map for all SeqLedger workflows and references.
- [CLI quickstart](docs/getting-started/cli-quickstart.md): first runnable path (`init` -> `import` -> `attach` -> `materialize` -> `export`).
- [Operations runbooks](docs/operations/README.md): remote sync, iterative batch loops, and transfer safety checks.
- [Reference index](docs/reference/README.md): stable contracts for schema, overlays/registry, event payloads, and API usage.
- [Architecture introspection](docs/architecture-introspection.md): package lifecycle, component boundaries, and interaction map.
- [Repository docs index](../../../docs/README.md): cross-tool workflows that connect SeqLedger with DenseGen, Infer, and Notify.

## Dataset location

SeqLedger uses `./seqledger_datasets` relative to the working directory by default. Create
that directory before starting a new collection, or select an existing root with
`--root /path/to/datasets` or `DNADESIGN_SEQLEDGER_ROOT`. Installation and upgrades do
not move datasets.

Existing dataset directories can be selected with `--root`; no relocation or
record conversion is required. Storage metadata, namespaced columns and lock
filenames remain stable.

## Remote configuration

Remote profiles are local operator configuration, not repository data. Copy
[`remotes.example.yaml`](remotes.example.yaml) to a private, untracked path,
replace the example values, and select it explicitly:

```bash
mkdir -p "$HOME/.config/dnadesign"
install -m 600 src/dnadesign/seqledger/remotes.example.yaml "$HOME/.config/dnadesign/seqledger-remotes.yaml"
# Edit seqledger-remotes.yaml and replace every example value before use.
export SEQLEDGER_REMOTES_PATH="$HOME/.config/dnadesign/seqledger-remotes.yaml"
uv run seqledger remotes list
```

The loader does not fall back to the tracked example or to a repository-local
active profile.
