# USR Python API quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-23

Public import surface: `dnadesign.usr`

Mutation methods require a registry at the dataset root.

Internal modules under `dnadesign.usr.src.*` are implementation details. Cross-tool callers should
import from `dnadesign.usr`; the old sibling root modules such as `dnadesign.usr.dataset` and
`dnadesign.usr.roots` are retired.

Within `dnadesign.usr.src`, root modules are reserved for coordinators. Helper families now live
under `cli_support/`, `datasets/`, `legacy/`, `overlay_support/`, `remote_sync/`, and `storage/`.
Remote sync execution orchestration now lives under `remote_sync/`, keeping `sync.py` as the
sanctioned root coordinator facade.
Low-level parquet IO, snapshotting, and dataset locking belong under `storage/`, not as sibling
root modules.
Internal helper families such as root/path resolution and schema/table presentation now live under
`cli_support/`, not as sibling root modules; the same applies to CLI-only stderr filtering.
Closed helper clusters should stay nested under those families, for example
`cli_commands/datasets/`, `cli_commands/lifecycle/`, `cli_commands/maintenance/`, `cli_commands/namespace/`, `cli_commands/query/`, `cli_commands/read_views/`,
`cli_commands/remotes/`, `cli_commands/sync/`, `cli_commands/tooling/`, `datasets/lifecycle/`, `datasets/merge/`, `datasets/overlay/`,
`datasets/query/`, `datasets/state/`, `datasets/validate/`, and `datasets/views/`.

Bootstrap example:

```bash
uv run usr --root src/dnadesign/usr/datasets namespace register mock \
  --columns 'mock__score:float64'
```

Python usage:

```python
from pathlib import Path
from dnadesign.usr import Dataset

root = Path("src/dnadesign/usr/datasets").resolve()

ds = Dataset.open(root, "densegen/demo_py")
ds.init(source="python quickstart")

result = ds.add_sequences(
    [{"sequence": "ACGTACGTAC"}],
    bio_type="dna",
    alphabet="dna_4",
    source="unit-test",
)
print(result.added)

overlay_df = ds.head(1, include_deleted=True)[["id"]].assign(mock__score=1.0)
ds.write_overlay("mock", overlay_df, key="id")

print(ds.head(3))
```

## Next steps

- Schema and registry contracts: [schema-contract.md](schema-contract.md), [overlay-and-registry.md](overlay-and-registry.md)
- Sync for cross-machine loops: [../operations/sync.md](../operations/sync.md)
