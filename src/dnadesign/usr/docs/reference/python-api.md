# USR Python API quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

Public import surface: `dnadesign.usr`

Mutation methods require a registry at the dataset root.
Registry namespace setup is part of the public library surface via
`parse_columns_spec` and `register_namespace`; cross-tool tests and operators
should not import `dnadesign.usr.src.registry` directly.

Internal modules under `dnadesign.usr.src.*` are implementation details. Cross-tool callers should
import from `dnadesign.usr`; the old sibling root modules such as `dnadesign.usr.dataset` and
`dnadesign.usr.roots` are retired.

Within `dnadesign.usr.src`, root packages are reserved for coordinators. Helper families now live
under `cli/support/`, `contracts/`, `datasets/`, `events/`, `legacy/`, `overlays/support/`,
`overlays/`, `regulondb/`, `sync/remote/`, `registry/`, `runtime/`, and `storage/`.
`usr/src` root should contain package directories only plus `__init__.py`; flat implementation
files at that level are considered an architecture regression.
Shared error/schema/type/sequence contracts now live under `contracts/`.
USR event logging helpers now live under `events/`, keeping the event import surface stable while
separating actor normalization, redaction, fingerprinting, and recording internals.
Overlay path/metadata helpers now live under `overlays/`, not as a sibling root helper module.
Registry loading, hashing, and validation now live under `registry/`, not as a sibling root
helper module.
Remote sync execution orchestration now lives under `sync/remote/`, keeping `sync/` as the
sanctioned coordinator package facade.
DuckDB session initialization and UTC enforcement now live under `runtime/`, not as a sibling
root helper.
Low-level parquet IO, snapshotting, and dataset locking belong under `storage/`, not as sibling
root modules.
Internal helper families such as root/path resolution, dependency/registration wiring, and
schema/table presentation now live under `cli/support/resolution/`,
`cli/support/wiring/`, and `cli/support/presentation/`, not as sibling root modules;
the same applies to CLI-only stderr filtering.
Ops-owned stable drill entrypoints live outside `dnadesign.usr.src`; use `uv run usr-sync-audit-drill`
for the deterministic sync audit drill instead of depending on raw script paths under
`src/dnadesign/usr/scripts/`.
Closed helper clusters should stay nested under those families, for example
`cli/commands/datasets/`, `cli/commands/lifecycle/`, `cli/commands/maintenance/`, `cli/commands/namespace/`, `cli/commands/query/`, `cli/commands/read_views/`,
`cli/commands/remotes/`, `cli/commands/sync/`, `cli/commands/tooling/`, `datasets/core/`, `datasets/demo/`,
`datasets/lifecycle/`, `datasets/maintenance/`, `datasets/merge/`, `datasets/overlay/`, `datasets/query/`,
`datasets/state/`, `datasets/validate/`, and `datasets/views/`.

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

ds = Dataset.open(root, "densegen_demo_py")
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
- Sync for cross-machine loops: [USR sync operations](../operations/sync/README.md)
