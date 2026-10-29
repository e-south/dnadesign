<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../README.md) / [Reference](README.md)

# Use SeqLedger from Python

Public import surface: `dnadesign.seqledger`. Use `Dataset` for sequence records and annotations, and the registry helpers to declare output types. The example below creates a local dataset; run it in an activated [SeqLedger environment](../getting-started/cli-quickstart.md#1-install-and-activate-seqledger).

## Create, annotate and read

```python
from pathlib import Path
from dnadesign.seqledger import Dataset, parse_columns_spec, register_namespace

# Use a new directory for this example.
root = Path("seqledger_python_demo").resolve()
root.mkdir()
register_namespace(
    root,
    namespace="demo",
    columns=parse_columns_spec("demo__score:float64", namespace="demo"),
)

ds = Dataset(root, "example_sequences")
ds.init(source="python quickstart")
result = ds.add_sequences(
    [{"sequence": "ACGTACGT"}],
    bio_type="dna",
    alphabet="dna_4",
    source="python quickstart",
)
print(result.added)  # 1 new record

# Join annotations by the ID assigned to the input sequence.
annotations = ds.head(1)[["id"]].assign(demo__score=0.8)
ds.write_overlay("demo", annotations, key="id")
print(ds.head(1)[["sequence", "demo__score"]].to_dict("records"))
```

Expected final output:

```text
[{'sequence': 'ACGTACGT', 'demo__score': 0.8}]
```

`head()` returns a pandas DataFrame with overlays included by default. To read a larger dataset in batches, use `scan()`:

```python
# Each batch is a PyArrow RecordBatch with effective overlay values.
for batch in ds.scan(columns=["id", "sequence", "demo__score"]):
    print(batch.num_rows)

ds.export("parquet", Path("annotated_sequences.parquet"))
```

Export reads the combined view and leaves the dataset unchanged. Reading `records.parquet` directly gives only the base table and can miss newer annotations.

## Write incremental batches

`write_overlay_part()` appends annotations without replacing a whole namespace:

```python
# Add a new sequence, then annotate only that record.
ds.add_sequences(
    [{"sequence": "TGCATGCA"}],
    bio_type="dna",
    alphabet="dna_4",
    source="next batch",
)
records = ds.head(2)
next_batch = records.loc[records["sequence"] == "TGCATGCA", ["id"]].assign(
    demo__score=0.5
)
ds.write_overlay_part("demo", next_batch, key="id", overwrite=False)
```

With `overwrite=False`, the occupied-cell check runs inside the dataset write lock. New rows, new columns and fills of null cells are allowed. A collision leaves data and mutation events unchanged. The default `overwrite=True` permits replacement; choose explicitly when producing incremental results.

## Read sequence views

Sequence views distinguish semantic uses of a sequence, such as an inserted region and a flanked construct. They are separate from annotation overlays. Use `load_sequence_views()` and `load_view_semantics()` from `dnadesign.seqledger` rather than reading the sidecar files directly. The [schema reference](schema-contract.md#sidecars-and-semantic-aliases) explains their identities and fields.

## Extend an integration

Import public APIs from `dnadesign.seqledger`; modules under `dnadesign.seqledger.src` are implementation details. The [architecture guide](../architecture-introspection.md) describes ownership and validation. For mutation rules, see [overlays and the registry](overlay-and-registry.md); for deliberate rewrites, see [maintenance](maintenance.md).
