# SeqLedger Python API quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28

Public import surface: `dnadesign.seqledger`

Mutation methods require a registry at the dataset root.
Registry namespace setup is part of the public library surface via
`parse_columns_spec` and `register_namespace`; cross-tool tests and operators
should not import `dnadesign.seqledger.src.registry` directly.

Import public APIs from `dnadesign.seqledger`. Modules under `dnadesign.seqledger.src`
are implementation details; see the [code map](dataset-layout-and-code-map.md)
for maintainer navigation.

Bootstrap example:

```bash
mkdir -p seqledger_datasets
uv run seqledger --root ./seqledger_datasets namespace register mock \
  --columns 'mock__score:float64'
```

Python usage:

```python
from pathlib import Path
from dnadesign.seqledger import Dataset

root = Path("seqledger_datasets").resolve()

ds = Dataset(root, "example_sequences")
ds.init(source="python quickstart")

result = ds.add_sequences(
    [{"sequence": "ACGTACGTAC"}],
    bio_type="dna",
    alphabet="dna_4",
    source="python quickstart",
)
print(result.added)

overlay_df = ds.head(1, include_deleted=True)[["id"]].assign(mock__score=1.0)
ds.write_overlay("mock", overlay_df, key="id")

print(ds.head(3))
```

For incremental producers, `ds.write_overlay_part("mock", batch, overwrite=False)`
checks occupied cells while holding the dataset write lock. It permits filling
nulls and adding disjoint rows or columns, but rejects a batch that targets an
existing non-null value. Rejected collisions leave data and mutation events
unchanged. The default `overwrite=True` retains last-writer-wins behavior.

## Next steps

- Schema and registry contracts: [schema-contract.md](schema-contract.md), [overlay-and-registry.md](overlay-and-registry.md)
- Deterministic sync check: `uv run seqledger-sync-audit-drill --help`
- Sync for cross-machine loops: [SeqLedger sync operations](../operations/sync/README.md)
