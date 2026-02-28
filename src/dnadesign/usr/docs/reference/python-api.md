# USR Python API quickstart

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


Mutation methods require a registry at the dataset root.

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
