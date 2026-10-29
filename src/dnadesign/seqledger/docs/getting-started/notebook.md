<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[SeqLedger guide](../README.md) / [Getting started](README.md)

# Explore a dataset in the notebook

The marimo explorer provides filters, summaries and plots for a selected Parquet table. It reads `records.parquet` directly, so it does not apply SeqLedger overlays or deletion filtering. To inspect the effective annotated view, first [export it](cli-quickstart.md#5-validate-and-export), then choose that file in **Custom parquet path**.

## Open the explorer

Run from a DNADesign checkout. The full profile includes marimo and the plotting dependencies used by this notebook.

```bash
# Install the notebook dependencies.
uv sync --locked

# Point the dataset picker at an existing root directory.
export DNADESIGN_SEQLEDGER_ROOT="/absolute/path/to/seqledger_datasets"

# Open the interactive notebook.
uv run marimo edit --watch src/dnadesign/seqledger/notebooks/usr_explorer.py
```

Replace the root path before running. The picker lists immediate child directories containing `records.parquet`; use **Custom parquet path** for nested datasets or an exported table. The sample slider limits the displayed rows, with a seed control for repeatable sampling.

## Inspect from the terminal

With SeqLedger installed, these commands provide a quicker view of a dataset:

```bash
# Include overlays by selecting a dataset through SeqLedger.
seqledger --root /absolute/path/to/seqledger_datasets head example_sequences
seqledger --root /absolute/path/to/seqledger_datasets schema example_sequences --tree

# Inspect a standalone exported table instead.
seqledger head ./annotated_sequences.parquet
seqledger cols ./annotated_sequences.parquet
seqledger cell ./annotated_sequences.parquet --row 0 --col sequence
```

Use the [Python API](../reference/python-api.md) for scripted analysis or the [schema reference](../reference/schema-contract.md) to interpret record and annotation fields.
