<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../README.md)

# Create your first dataset

Import three DNA sequences, attach example scores and export the annotated table. Everything stays in a new local directory; no remote service or model is needed.

## 1. Install and activate SeqLedger

From a DNADesign checkout:

```bash
uv sync --locked --no-default-groups --extra seqledger
source .venv/bin/activate
```

If SeqLedger is already installed, activate that environment instead. The remaining commands use its `seqledger` executable and can run outside the checkout. See [installation](../../../../../docs/setup/installation.md) for other profiles.

## 2. Prepare a small input

Run this where you want to keep the example:

```bash
mkdir seqledger-demo
cd seqledger-demo
mkdir seqledger_datasets
DATA_ROOT="$PWD/seqledger_datasets"
DATASET="example_sequences"

cat > sequences.csv <<'CSV'
sequence
ACGTACGT
ACGTTCGT
TGCATGCA
CSV
```

A dataset root can contain several datasets and one shared `registry.yaml`. Each command below names the root explicitly with `--root`.

## 3. Register an annotation and import records

A namespace declares which annotation columns you will write. Register `demo__score` as a floating-point value; this also creates the root's registry.

```bash
seqledger --root "$DATA_ROOT" namespace register demo \
  --columns 'demo__score:float64'
seqledger --root "$DATA_ROOT" init "$DATASET" --source "quickstart"
seqledger --root "$DATA_ROOT" import "$DATASET" --from csv \
  --path sequences.csv --bio-type dna --alphabet dna_4
seqledger --root "$DATA_ROOT" head "$DATASET" -n 3
```

The dataset now contains three records. SeqLedger computes their IDs and adds biological type, alphabet, length, source and creation time. Sequence case contributes to identity; the [schema reference](../reference/schema-contract.md) explains normalization and duplicate handling.

## 4. Attach scores

Create a second file that pairs the same sequences with illustrative scores:

```bash
cat > scores.csv <<'CSV'
sequence,score
ACGTACGT,0.2
ACGTTCGT,0.8
TGCATGCA,0.5
CSV

# Match by sequence and store the score in the demo namespace.
seqledger --root "$DATA_ROOT" attach "$DATASET" \
  --path scores.csv --namespace demo \
  --key sequence --key-col sequence --columns score
seqledger --root "$DATA_ROOT" head "$DATASET" -n 3
```

The combined view includes `demo__score`. These values live in an overlay under `_derived/`; the base `records.parquet` is not rewritten. For other join keys or updates to existing annotations, see [overlays and the registry](../reference/overlay-and-registry.md).

## 5. Validate and export

```bash
# Check the dataset and its registered annotation types.
seqledger --root "$DATA_ROOT" validate "$DATASET" --strict

# Export the combined view to an explicit file path.
seqledger --root "$DATA_ROOT" export "$DATASET" --fmt csv \
  --columns sequence,demo__score --out annotated_sequences.csv
cat annotated_sequences.csv
```

The CSV pairs each sequence with its score:

```csv
sequence,demo__score
ACGTACGT,0.2
ACGTTCGT,0.8
TGCATGCA,0.5
```

Export includes overlays without materialization. It produces a table for another program, not a full copy of dataset history or sidecars. Use [dataset sync](../operations/sync/README.md) when transferring the complete dataset.

## Inspect or continue

```bash
# List datasets, inspect the schema and read recent mutation events.
seqledger --root "$DATA_ROOT" ls
seqledger --root "$DATA_ROOT" schema "$DATASET" --tree
seqledger --root "$DATA_ROOT" events tail "$DATASET" --format json --n 5
```

- [Python API](../reference/python-api.md): perform the same work in a script.
- [Notebook](notebook.md): browse distributions and columns interactively.
- [Maintenance](../reference/maintenance.md): snapshots, compaction and deliberate base-table rewrites.
